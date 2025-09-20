from typing import Union


import time
import dataclasses
from dataclasses import dataclass, field
from typing import Type, Tuple
from typing_extensions import Literal
from copy import deepcopy
import os
from pathlib import Path
from shutil import copyfile
from scipy.spatial.transform import Rotation

import torch
from torch import Tensor
import rclpy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Path as NavPath
from geometry_msgs.msg import PoseStamped

from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.utils import profiler, writer
from nerfstudio.engine.callbacks import (
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
    TrainingCallback,
)

from nerfbridge.ros_dataset import ROSDataset
from nerfbridge.ros_viewer import ROSViewer
from nerfbridge.writer import CONSOLE
from nerfbridge.pointcloud_utils import make_point_cloud

# Suppress nerfstudio console output
from nerfstudio.utils.rich_utils import CONSOLE as NS_CONSOLE

NS_CONSOLE.quiet = True


@dataclass
class ROSTrainerConfig(TrainerConfig):
    _target: Type = field(default_factory=lambda: ROSTrainer)
    msg_timeout: float = 300.0
    """ How long to wait (seconds) for sufficient images to be received before training. """
    num_msgs_to_start: int = 5
    """ Number of images that must be recieved before training can start. """
    show_robot_in_viewer: bool = False
    """ Whether to show the robot state in the viewer. """
    publish_pcd: bool = False
    """ Whether to publish a point cloud to ROS. """
    publish_pcd_every: int = 500
    """ Training steps between publishing the pcd."""
    pcd_render_downscale: float = 0.5
    """ Camera resolution rescaling factor for PCD rendering. 0.5 means half resolution. """


class ROSTrainer(Trainer):
    config: ROSTrainerConfig
    dataset: ROSDataset

    def __init__(
        self, config: ROSTrainerConfig, local_rank: int = 0, world_size: int = 0
    ):
        # We'll see if this throws and error (it expects a different config type)
        super().__init__(config, local_rank=local_rank, world_size=world_size)
        self.msg_timeout = self.config.msg_timeout
        self.cameras_drawn = []
        self.num_msgs_to_start = config.num_msgs_to_start

        # Init ROS
        rclpy.init(args=None)

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datasets into memory
                'inference': does not load any dataset into memory
        """
        # Create simlink to the latest run
        symlink_path = self.config.output_dir.resolve() / "latest_run"
        if symlink_path.exists():
            os.remove(symlink_path)
        os.symlink(self.base_dir.resolve(), symlink_path, target_is_directory=True)

        self.pipeline = self.config.pipeline.setup(
            device=self.device,
            test_mode=test_mode,
            world_size=self.world_size,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
            cache_dir=self.base_dir / "run_data",
        )
        self.optimizers = self.setup_optimizers()

        # save the config file for debugging and later eval
        copyfile(
            self.pipeline.datamanager.get_datapath(),
            self.base_dir / "dataparser_config.json",
        )
        # set up viewer if enabled
        viewer_log_path = self.base_dir / self.config.viewer.relative_log_filename
        self.viewer_state, banner_messages = None, None
        if self.config.is_viewer_legacy_enabled() and self.local_rank == 0:
            CONSOLE.print(
                "[bold red] (NerfBridge) Legacy Viewer is not supported by NerfBridge!"
            )
        if self.config.is_viewer_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = ROSViewer(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
                share=self.config.viewer.make_share_url,
            )
            banner_messages = self.viewer_state.viewer_info

        self._check_viewer_warnings()

        # self._load_checkpoint()

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers,
                grad_scaler=self.grad_scaler,
                pipeline=self.pipeline,
                trainer=self,
            )
        )

        # set up writers/profilers if enabled
        writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        writer.setup_event_writer(
            self.config.is_wandb_enabled(),
            self.config.is_tensorboard_enabled(),
            self.config.is_comet_enabled(),
            log_dir=writer_log_path,
            experiment_name=self.config.experiment_name,
            project_name=self.config.project_name,
        )
        writer.setup_local_writer(
            self.config.logging,
            max_iter=self.config.max_num_iterations,
            banner_messages=banner_messages,
        )
        writer.put_config(
            name="config", config_dict=dataclasses.asdict(self.config), step=0
        )
        profiler.setup_profiler(self.config.logging, writer_log_path)

        if self.config.show_robot_in_viewer and self.config.is_viewer_enabled():
            pose_topic = self.pipeline.datamanager.train_dataset.pose_topic_name
            slam_method = self.pipeline.datamanager.train_image_dataloader.slam_method
            self.viewer_state.setup_robot_viewer_node(pose_topic, slam_method)
            self.pipeline.datamanager.train_image_dataloader.executor.add_node(
                self.viewer_state.node
            )

        if self.config.publish_pcd:
            self.setup_pcd_publisher()
            pcd_callback = TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.generate_and_publish_pcd,
                update_every_num_iters=self.config.publish_pcd_every,
            )
            self.callbacks.append(pcd_callback)

        # # Start a thread for processing all of the ros callbacks
        self.pipeline.datamanager.train_image_dataloader.start_ros_thread()

        # Start Status check loop
        start = time.perf_counter()
        status = False
        CONSOLE.print(
            f"[bold green] (NerfBridge) Waiting to recieve {self.num_msgs_to_start} images..."
        )
        # with CONSOLE.status("", spinner="dots") as status:
        while time.perf_counter() - start < self.msg_timeout:
            dl = self.pipeline.datamanager.train_image_dataloader
            dl.update_dataset_from_buffer(0)
            dl_idx = dl.buffered_idx
            if dl_idx >= (self.num_msgs_to_start - 1):
                status = True
                break
            else:
                status_str = f"[green] (NerfBridge) Images received: {dl_idx}"
                # status.update(status_str)
            time.sleep(0.05)

        self.dataset = self.pipeline.datamanager.train_dataset  # pyright: ignore

        if not status:
            raise NameError(
                "(NerfBridge) ROSTrainer setup() timed out, check that messages \
                are being published and that config.json correctly specifies topic names."
            )
        else:
            CONSOLE.print(
                "[bold green] (NerfBridge) Pre-train image buffer filled, starting training!"
            )

    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers

        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Data states are not included in module state_dict for pipeline so collect
        # them and add them to the checkpoint seperately
        data_states = {}
        idx = self.pipeline.datamanager.train_image_dataloader.buffered_idx
        data_states["buffered_idx"] = idx
        data_states["train_c2ws"] = self.pipeline.datamanager.train_dataset.cameras.camera_to_worlds
        data_states["camera_K"] = self.pipeline.datamanager.train_dataset.K
        data_states["camera_distorts"] = self.pipeline.datamanager.train_dataset.dparams
        data_states["camera_H"] = self.pipeline.datamanager.train_dataset.image_height
        data_states["camera_W"] = self.pipeline.datamanager.train_dataset.image_width
        # data_states["eval_c2ws"] = torch.stack(self.pipeline.datamanager.train_image_dataloader.eval_poses)

        # save the checkpoint
        ckpt_path: Path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else self.pipeline.state_dict(),
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "schedulers": {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()},
                "scalers": self.grad_scaler.state_dict(),
                "data_states": data_states,
            },
            ckpt_path,
        )
        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("*"):
                if f != ckpt_path:
                    f.unlink()

    def setup_pcd_publisher(self):
        """Setup the PCD publisher."""
        self.pcd_node = rclpy.create_node("nerfbridge_pcd_publisher")
        self.pcd_publisher = self.pcd_node.create_publisher(
            PointCloud2, "/nerfbridge/pcd_gsplat", 10
        )

        self.training_pose_pub = self.pcd_node.create_publisher(
                    NavPath, "/nerfbridge/training_pose", 10)
            
        self.goal_str = "trashcan"
        self.negative_str = "stuff, objects, things"

    def compute_similarity(self, outputs):
        # Normalize CLIP features rendered by feature field
        clip_features = outputs["feature"]
        clip_features /= clip_features.norm(dim=-1, keepdim=True)

        # If there are no negatives, just show the cosine similarity with the positives
        if not self.pipeline.model.viewer_utils.has_negatives:
            sims = clip_features @ self.viewer_utils.pos_embed.T
            # Show the mean similarity if there are multiple positives
            if sims.shape[-1] > 1:
                sims = sims.mean(dim=-1, keepdim=True)
            return sims

        # Use paired softmax method as described in the paper with positive and negative texts
        text_embs = torch.cat(
            [
                self.pipeline.model.viewer_utils.pos_embed,
                self.pipeline.model.viewer_utils.neg_embed,
            ],
            dim=0,
        )
        raw_sims = clip_features @ text_embs.T

        # Broadcast positive label similarities to all negative labels
        pos_sims, neg_sims = raw_sims[..., :1], raw_sims[..., 1:]
        pos_sims = pos_sims.broadcast_to(neg_sims.shape)
        paired_sims = torch.cat([pos_sims, neg_sims], dim=-1)

        # Compute paired softmax
        probs = (paired_sims / self.pipeline.model.viewer_utils.softmax_temp).softmax(
            dim=-1
        )[..., :1]
        torch.nan_to_num_(probs, nan=0.0)
        sims, _ = probs.min(dim=-1, keepdim=True)
        # outputs["similarity"] = sims
        return sims

    def generate_pcd_nerfacto(
        self, num_points: int = 100000
    ) -> Tuple[PointCloud2, Tensor]:
        """Generate a point cloud with relevancies from a DFF.

        Args:
            num_points: Number of points to generate. May result in less if
                outlier removal is used.

        Returns:
            Point cloud.
        """
        points = []
        rgbs = []
        similarities = []
        origins = []
        color_errs = []

        # Set the semantic queries for the goals
        self.pipeline.model.viewer_utils.handle_language_queries(
            raw_text=self.goal_str, is_positive=True
        )
        self.pipeline.model.viewer_utils.handle_language_queries(
            raw_text=self.negative_str, is_positive=False
        )

        curr_points = 0
        while curr_points < num_points:
            with torch.no_grad():
                ray_bundle, pixels = self.pipeline.datamanager.next_train(0)
                outputs = self.pipeline.model(ray_bundle)
                color_err = torch.norm(outputs["rgb"] - pixels["image"].cuda(), dim=-1)
            rgba = self.pipeline.model.get_rgba_image(outputs, "rgb")
            depth = outputs["depth"]
            similarity = self.compute_similarity(outputs)

            # termination points
            point = ray_bundle.origins + ray_bundle.directions * depth
            mask = rgba[..., -1] > 0.5  # TODO: tune this threshold?
            origin = ray_bundle.origins[mask]
            rgb = rgba[mask][..., :3]
            point = point[mask]
            similarity = similarity[mask]
            color_err = color_err[mask]

            curr_points += len(point)
            points.append(point)
            rgbs.append(rgb)
            similarities.append(similarity)
            origins.append(origin)
            color_errs.append(color_err.reshape(-1, 1))

        points = torch.cat(points, dim=0)
        origins = torch.cat(origins, dim=0)

        # Map points back to odom frame
        scene_scale = self.pipeline.datamanager.train_dataset.scene_scale_factor
        center_T = self.pipeline.datamanager.train_image_dataloader.zero_pose
        if center_T is None:
            center_T = torch.eye(4).to(device=points.device)
        inv_center_T = torch.inverse(center_T).to(device=points.device)
        points = points / scene_scale
        points = points @ inv_center_T[:3, :3].T + inv_center_T[:3, 3]
        origins = origins / scene_scale
        origins = origins @ inv_center_T[:3, :3].T + inv_center_T[:3, 3]

        rgbs = torch.cat(rgbs, dim=0).cpu()
        points = points.cpu()
        origins = origins.cpu()
        similarities = torch.cat(similarities, dim=0).cpu()
        color_errs = torch.cat(color_errs, dim=0).cpu()
        torch_pcd = torch.hstack((points, rgbs, similarities, origins, color_errs))

        rgbs = (rgbs * 255).to(dtype=torch.uint8, device="cpu")
        pcd_msg = make_point_cloud(
            points=points,
            colors=rgbs,
            sims=similarities,
            origins=origins,
            frame_id="odom",
        )
        return pcd_msg, torch_pcd

    def generate_pcd_gsplat(
        self, num_points: int = 100000
    ) -> Tuple[PointCloud2, Tensor]:
        """Generate relevant point cloud values from the checkpoint data states.
        This function is for exporting to point cloud then mesh.
        """

        points = []
        colors = []
        similarities = []
        origins = []

        # Set semantic query from configuration json
        pos_query = self.pipeline.datamanager.train_dataset.metadata[
            "semantics_positives"
        ]
        neg_query = self.pipeline.datamanager.train_dataset.metadata[
            "semantics_negatives"
        ]

        self.pipeline.model.viewer_utils.handle_language_queries(
            ", ".join(pos_query), True
        )
        self.pipeline.model.viewer_utils.handle_language_queries(
            ", ".join(neg_query), False
        )

        current_num_cams = self.pipeline.datamanager.train_image_dataloader.buffered_idx
        points_per_cam = num_points // current_num_cams

        for idx in range(current_num_cams):
            camera = self.pipeline.datamanager.train_dataset.cameras[idx : idx + 1]
            camera = deepcopy(camera)  # Avoid modifying the original camera

            # Rescale the output resolution to speed up rendering
            camera.rescale_output_resolution(self.config.pcd_render_downscale)
            c2w = camera.camera_to_worlds[0]

            self.pipeline.eval()
            with torch.no_grad():
                outputs = self.pipeline.model.get_outputs(camera)
                depth = outputs["depth"]
                pcd = self.pipeline.model.get_point_cloud_from_camera(camera, depth)
            self.pipeline.train()

            rand_idxs = torch.randperm(pcd.view(-1, 3).shape[0], device=pcd.device)[
                :points_per_cam
            ]
            xyzs = pcd.view(-1, 3)[rand_idxs]
            c2w = c2w.squeeze()

            rgbs = outputs["rgb"].view(-1, 3)[rand_idxs]
            sims = outputs["similarity"].view(-1)[rand_idxs]
            origin = c2w[:3, 3].unsqueeze(0).expand(xyzs.shape)

            points.append(xyzs.cpu())
            colors.append(rgbs.cpu())
            similarities.append(sims.cpu())
            origins.append(origin.cpu())

        points = torch.cat(points, dim=0)
        colors = torch.cat(colors, dim=0)
        similarities = torch.cat(similarities, dim=0).unsqueeze(1)
        origins = torch.cat(origins, dim=0)

        # Map points back to odom frame
        scene_scale = self.pipeline.datamanager.train_dataset.scene_scale_factor
        center_T = self.pipeline.datamanager.train_image_dataloader.zero_pose
        if center_T is None:
            center_T = torch.eye(4).to(device=points.device)
        inv_center_T = torch.inverse(center_T).to(device=points.device)

        points = points / scene_scale
        points = points @ inv_center_T[:3, :3].T + inv_center_T[:3, 3]
        origins = origins / scene_scale
        origins = origins @ inv_center_T[:3, :3].T + inv_center_T[:3, 3]

        torch_pcd = torch.hstack((points, colors, similarities, origins))

        rgbs = (colors * 255).to(dtype=torch.uint8, device="cpu")
        pcd_msg = make_point_cloud(
            points=points,
            colors=rgbs,
            sims=similarities,
            origins=origins,
            frame_id="map",
        )
        return pcd_msg, torch_pcd
    
    @torch.no_grad()
    def generate_means_gsplat(
        self
    ) -> Tuple[PointCloud2, Tensor]:
        """Generate relevant point cloud values from the checkpoint data states.
        This function is for exporting to point cloud then mesh.
        """

        # points = []
        # colors = []
        # similarities = []
        # origins = []
        start_time = time.time()
        # Set semantic query from configuration json
        pos_query = self.pipeline.datamanager.train_dataset.metadata[
            "semantics_positives"
        ]
        neg_query = self.pipeline.datamanager.train_dataset.metadata[
            "semantics_negatives"
        ]

        self.pipeline.model.viewer_utils.handle_language_queries(
            ", ".join(pos_query), True
        )
        self.pipeline.model.viewer_utils.handle_language_queries(
            ", ".join(neg_query), False
        )
        viewer_time = time.time()
        points = self.pipeline.model.means.clone().detach()
        colors = torch.sigmoid(self.pipeline.model.features_dc[:, :].clone().detach())
        means_time = time.time()
        ## sim stuff ##
        clip_features = self.pipeline.model.clip_field(points*self.pipeline.model.config.semantic_field_scale).float()
        clip_features /= clip_features.norm(dim=-1, keepdim=True)

        if self.pipeline.model.viewer_utils.has_positives:
            if self.pipeline.model.viewer_utils.has_negatives:
                # Use paired softmax method as described in the paper with positive and negative texts
                text_embs = torch.cat(
                    [self.pipeline.model.viewer_utils.pos_embed, self.pipeline.model.viewer_utils.neg_embed],
                    dim=0,
                )

                raw_sims = clip_features @ text_embs.T

                # Broadcast positive label similarities to all negative labels
                pos_sims, neg_sims = raw_sims[..., :1], raw_sims[..., 1:]
                pos_sims = pos_sims.broadcast_to(neg_sims.shape)

                # Updated Code
                paired_sims = torch.cat(
                    (pos_sims.reshape((-1, 1)), neg_sims.reshape((-1, 1))), dim=-1
                )

                # compute the paired softmax
                probs = paired_sims.softmax(dim=-1)[..., :1]
                probs = probs.reshape((-1, neg_sims.shape[-1]))

                torch.nan_to_num_(probs, nan=0.0)

                sims, _ = probs.min(dim=-1, keepdim=True)
                sims = sims.reshape((*pos_sims.shape[:-1], 1))

            else:
                # positive embeddings
                text_embs = self.viewer_utils.pos_embed

                sims = clip_features @ text_embs.T
                # Show the mean similarity if there are multiple positives
                if sims.shape[-1] > 1:
                    sims = sims.mean(dim=-1, keepdim=True)
        sims_time = time.time()
        # origins = torch.zeros((sims.shape[0], 3)).to(points.device)

        # Map points back to odom frame
        scene_scale = self.pipeline.datamanager.train_dataset.scene_scale_factor
        center_T = self.pipeline.datamanager.train_image_dataloader.zero_pose
        if center_T is None:
            center_T = torch.eye(4).to(device=points.device)
        inv_center_T = torch.inverse(center_T).to(device=points.device)

        points = points / scene_scale
        points = points @ inv_center_T[:3, :3].T + inv_center_T[:3, 3]

        rgbs = (colors * 255).to(dtype=torch.uint8, device="cpu")
        math_time = time.time()

        # add deltas
        try:
            deltas = self.pipeline.model.param_delta.clone().detach().reshape((-1,1))
            torch_pcd = torch.hstack((points, colors, sims, deltas))
            pcd_msg = make_point_cloud(
                points=points.cpu().numpy(),
                colors=rgbs.cpu().numpy(),
                sims=sims.cpu().numpy(),
                # origins=origins,
                deltas=deltas.cpu().numpy(),
                frame_id="map",
            )
        except:
            torch_pcd = torch.hstack((points, colors, sims))
            pcd_msg = make_point_cloud(
                points=points.cpu().numpy(),
                colors=rgbs.cpu().numpy(),
                sims=sims.cpu().numpy(),
                # origins=origins,
                frame_id="map",
            )
        
        pcd_msg_time = time.time()
        return pcd_msg, torch_pcd

    def generate_and_publish_pcd(self, step: int):
        """Publish a PCD to ROS."""
        # pcd_msg, pcd = self.generate_pcd_gsplat()
        start_time = time.time()
        pcd_msg, pcd = self.generate_means_gsplat()
        pcd_time = time.time()
        # print("pcd_msg: ", pcd_msg)
        if pcd_msg is not None:
            
            self.pcd_publisher.publish(pcd_msg)

            current_num_cams = self.pipeline.datamanager.train_image_dataloader.buffered_idx
            # print("current_num_cams: ", current_num_cams)
            # print("num cameras: ", len(self.pipeline.datamanager.train_dataset.cameras))
            training_poses_msg = NavPath()
            c2ws = []
            num_sampled_cameras = 200
            rand_idxs = torch.randperm(current_num_cams, device=pcd.device)[
                :num_sampled_cameras
            ]
            for idx in rand_idxs:
                camera = self.pipeline.datamanager.train_dataset.cameras[idx : idx + 1]
                camera = deepcopy(camera)  # Avoid modifying the original camera

                # Rescale the output resolution to speed up rendering
                camera.rescale_output_resolution(self.config.pcd_render_downscale)
                c2w = torch.eye(4, device=pcd.device)
                c2w[0:3,0:4] = camera.camera_to_worlds[0]
                c2ws.append(c2w)
                rot = Rotation.from_matrix(c2w[0:3, 0:3].cpu().numpy())
                quat = rot.as_quat()
                pose_msg = PoseStamped()
                pose_msg.pose.position.x = float(c2w[0,3])
                pose_msg.pose.position.y = float(c2w[1,3])
                pose_msg.pose.position.z = float(c2w[2,3])

                pose_msg.pose.orientation.x = quat[0]
                pose_msg.pose.orientation.y = quat[1]
                pose_msg.pose.orientation.z = quat[2]
                pose_msg.pose.orientation.w = quat[3]

                training_poses_msg.poses.append(pose_msg)

            self.training_pose_pub.publish(training_poses_msg)
            c2ws = torch.stack(c2ws, dim=0)
        if not (self.base_dir / "pointclouds").exists():
            (self.base_dir / "pointclouds").mkdir()
            (self.base_dir / "training_poses").mkdir()
        if pcd is not None:
            torch.save(pcd, self.base_dir / "pointclouds" / f"pcd_{step}.pt")
            torch.save(c2ws, self.base_dir / "training_poses" / f"poses_{step}.pt")
