# Code adapted from Nerfstudio

# https://github.com/nerfstudio-project/nerfstudio/blob/df784e96e7979aaa4320284c087d7036dce67c28/nerfstudio/data/utils/dataloaders.py

"""
Defines the ROSDataloader object that subscribes to pose and images topics,
and populates an image tensor and Cameras object with values from these topics.
Image and pose pairs are added at a prescribed frequency and intermediary images
are discarded (could be used for evaluation down the line).
"""

import time
import threading
import warnings
from typing import Union, Tuple, Optional
from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import cv2
import open3d as o3d
import numpy as np

import nerfbridge.pose_utils as pose_utils
from nerfbridge.pointcloud_utils import read_points
from nerfbridge.ros_dataset import ROSDataset
from nerfbridge.features.clip_wrapper import CLIPWrapper
from nerfbridge.writer import DataStatPrinter


# Suppress a warning from torch.tensorbuffer about copying that
# does not apply in this case.
warnings.filterwarnings("ignore", "The given buffer")


class ROSDataloader(DataLoader):
    dataset: ROSDataset

    def __init__(
        self,
        dataset: ROSDataset,
        use_semantics: bool,
        cache_dir: Path,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        # This is mostly a parameter placeholder, and manages the cameras
        self.dataset = dataset
        self.use_semantics = use_semantics
        self.system = self.dataset.metadata["system"]

        if cache_dir is not None:
            self.train_cache_dir = cache_dir / "train"
            self.train_cache_dir.mkdir(parents=True)
            (self.train_cache_dir / "rgb").mkdir()
            (self.train_cache_dir / "depth").mkdir()
            if use_semantics:
                (self.train_cache_dir / "semantics").mkdir()

            self.eval_cache_dir = cache_dir / "eval"
            self.eval_cache_dir.mkdir()
            (self.eval_cache_dir / "rgb").mkdir()
            (self.eval_cache_dir / "depth").mkdir()
            # For now we don't save semantics to save space/time

        # Image meta data
        self.device = device
        self.num_images = len(self.dataset)
        self.H = self.dataset.image_height
        self.W = self.dataset.image_width
        self.image_crops = self.dataset.metadata["image_crops"]
        self.n_channels = 3
        self.pose_reject_radius = self.dataset.metadata["pose_reject_radius"]

        # Tracking ros updates
        self.current_idx = 0
        self.buffered_idx = 0
        self.updated = True
        self.update_period = 1 / self.dataset.metadata["data_update_freq"]

        # Eval dataset stuff
        self.eval_period = 1 / self.dataset.metadata["eval_save_freq"]
        self.eval_idx = 0
        self.eval_poses = []

        self.pose_buffer = []

        # Keep it in the format so that it makes it look more like a
        # regular data loader.
        self.data_dict = {
            "image": self.dataset.image_tensor,
            "image_idx": self.dataset.image_indices,
        }
        if self.system == "spot" or self.system == "zed":
            self.data_dict["depth_image"] = self.dataset.depth_tensor

        if use_semantics:
            self.sem_model = CLIPWrapper(self.H, self.W, device="cuda")
            self.dataset.semantics_tensor = torch.zeros(
                self.num_images,
                self.sem_model.out_h,
                self.sem_model.out_w,
                self.sem_model.feature_dim,
                dtype=torch.float32,
            )
            self.dataset.metadata["feature_type"] = self.sem_model.feature_type
            self.dataset.metadata["feature_dim"] = self.sem_model.feature_dim
            self.scale_h = self.sem_model.out_h / self.H
            self.scale_w = self.sem_model.out_w / self.W

        super().__init__(dataset=dataset, **kwargs)

        self.bridge = CvBridge()
        self.use_compressed_rgb = self.dataset.metadata["compressed_rgb"]
        self.undistort_images = self.dataset.metadata["undistort_rgb"]
        self.topic_sync = self.dataset.metadata["topic_sync"]
        self.topic_slop = self.dataset.metadata["topic_slop"]

        self.K = self.dataset.K.numpy()
        self.dparams = self.dataset.dparams.numpy()

        # Initializing ROS2
        self.node = rclpy.create_node("nerf_bridge_node")
        self.frame_names = self.dataset.metadata["camera_frame_names"]
        self.static_transforms = self.dataset.metadata["static_transforms"]
        self.last_pose = {}

        if self.system == "spot":
            # Spot Initialization
            self.subs = {name: [] for name in self.frame_names}

            for name, topic in zip(
                self.frame_names,
                self.dataset.image_topic_name,
            ):
                self.subs[name].append(
                    Subscriber(
                        self.node,
                        CompressedImage if self.use_compressed_rgb else Image,
                        topic,
                    )
                )
                self.subs[name].append(
                    Subscriber(self.node, Odometry, self.dataset.pose_topic_name)
                )

            for name, topic in zip(
                self.frame_names,
                self.dataset.depth_topic_name,
            ):
                self.subs[name].append(Subscriber(self.node, Image, topic))

            if self.topic_sync == "approximate":
                self.ts = {
                    name: ApproximateTimeSynchronizer(subs, 40, self.topic_slop)
                    for name, subs in self.subs.items()
                }
            else:
                self.ts = {
                    name: TimeSynchronizer(subs, 40) for name, subs in self.subs.items()
                }

            for ts in self.ts.values():
                ts.registerCallback(self.ts_callback)
        elif self.system == "modal":
            # Modal Initialization
            qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.subs = [
                Subscriber(
                    self.node,
                    CompressedImage if self.use_compressed_rgb else Image,
                    self.dataset.image_topic_name[0],
                ),
                Subscriber(self.node, PoseStamped, self.dataset.pose_topic_name, qos_profile=qos_profile),
                Subscriber(self.node, PointCloud2, self.dataset.depth_topic_name[0], qos_profile=qos_profile),
            ]

            if self.topic_sync == "approximate":
                self.ts = ApproximateTimeSynchronizer(self.subs, 40, self.topic_slop)
            else:
                self.ts = TimeSynchronizer(self.subs, 40)

            self.ts.registerCallback(self.ts_callback)
        elif self.system == "zed":
            qos_profile = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability = DurabilityPolicy.VOLATILE)
            self.subs = [
                Subscriber(
                    self.node,
                    CompressedImage if self.use_compressed_rgb else Image,
                    self.dataset.image_topic_name[0],
                ),
                Subscriber(self.node, PoseStamped, self.dataset.pose_topic_name, qos_profile=qos_profile),
                Subscriber(self.node, Image, self.dataset.depth_topic_name[0]),
            ]
            
            if self.topic_sync == "approximate":
                self.ts = ApproximateTimeSynchronizer(self.subs, 40, self.topic_slop)
            else:
                self.ts = TimeSynchronizer(self.subs, 40)

            self.ts.registerCallback(self.ts_callback)

        # Set update time
        now = time.perf_counter()
        self.last_update_t = {name: now for name in self.frame_names}
        self.last_eval_t = now

        self.center_to_first = self.dataset.metadata["center_to_first_pose"]
        self.zero_pose = None

        # Create a multi-threaded executor to spin
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)

        self.stat_printer = DataStatPrinter()

    def start_ros_thread(self):
        self.ros_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.ros_thread.start()

    def ts_callback(self, *args):
        if self.system == "spot":
            frame_name = args[0].header.frame_id
        else:
            frame_name = "pose_to_camera"
        last_t = self.last_update_t[frame_name]

        now = time.perf_counter()
        if now - last_t > self.update_period and self.current_idx < self.num_images:
            # Process RGB
            im_success, rgb = self.image_callback(args[0])
            success = im_success

            # Process Pose
            pose_success, pose = self.pose_callback(args[1], frame_name=frame_name)
            success = success and pose_success

            # Process Depth
            if self.system == "spot" or self.system == "zed":
                d_success, depth = self.depth_callback(args[2])
            elif self.system == "modal":
                depth = self.pointcloud_callback(args[2])
                d_success = True
            else:
                d_success = False
                depth = None

            success = success and d_success

            if success:
                # Insert image
                self.dataset.image_tensor[self.current_idx] = rgb
                # Add pose to buffer instead
                self.pose_buffer.append((self.current_idx, pose))

                if self.system == "spot" or self.system == "zed":
                    # Insert depth
                    self.dataset.depth_tensor[self.current_idx] = depth
                else:
                    # Insert pointcloud (MODAL)
                    self.dataset.pc_dict[self.current_idx] = depth

                # Compute and insert semantics
                if self.use_semantics:
                    features = self.sem_model.get_features(rgb)
                    self.dataset.semantics_tensor[self.current_idx] = features
                else:
                    features = None

                self.stat_printer.data["num_train"] += 1
                # self.stat_printer.print_table()

                # Save training data
                depth_save = None if self.system == "modal" else depth
                self.save_images(
                    self.train_cache_dir, self.current_idx, rgb, depth_save, features
                )

                self.updated = True
                self.current_idx += 1
                self.last_update_t[frame_name] = now
        else:
            # If we are not using the frame for training, we can use it for evaluation
            if (
                now - self.last_eval_t > self.eval_period
                and self.current_idx < self.num_images
            ):
                # Convert eval data to tensors and save.
                # Use the same callbacks to ensure consistency.

                # Process RGB
                _, rgb = self.image_callback(args[0])

                # Process Pose
                _, pose = self.pose_callback(
                    args[1], frame_name=frame_name, is_eval=True
                )

                # Process Depth
                if self.system == "spot" or self.system == "zed":
                    _, depth = self.depth_callback(args[2])
                elif self.system == "modal":
                    depth = None

                self.eval_poses.append(pose)
                self.save_images(self.eval_cache_dir, self.eval_idx, rgb, depth)
                self.eval_idx += 1
                self.last_eval_t = now

    def save_images(
        self,
        base_dir: Path,
        idx: int,
        image: torch.Tensor,
        depth: Optional[torch.Tensor],
        semantics: Optional[torch.Tensor] = None,
    ):
        """
        Saving of various images for debugging and evaluation.

        Pose information is saved in ros_trainer.py because we need to
        save the pose corrections and other parameters too.
        """
        im_cv = cv2.cvtColor(image.cpu().numpy(), cv2.COLOR_RGB2BGR)
        im_cv = (im_cv * 255).astype("uint8")

        image_path = base_dir / "rgb" / f"{idx:04d}.png"

        # Save rgb image
        cv2.imwrite(str(image_path), im_cv)

        # Save depth as torch tensor (this just makes saving and loading easier
        # because dealing with depth images in opencv is a pain)
        if depth is not None:
            depth_path = base_dir / "depth" / f"{idx:04d}.pt"
            depth = depth.squeeze().cpu().numpy()
            torch.save(depth, depth_path)

        # Save semantics if available
        if semantics is not None:
            semantics_path = base_dir / "semantics" / f"{idx:04d}.pt"
            torch.save(semantics, semantics_path)

    def image_callback(
        self, image: Union[Image, CompressedImage]
    ) -> Tuple[bool, torch.Tensor]:
        """
        Callback for processing RGB Image Messages, and adding them to the
        dataset for training.
        """
        # Load the image message directly into the torch
        if isinstance(image, CompressedImage):
            im_cv = self.bridge.compressed_imgmsg_to_cv2(image)
        else:
            im_cv = self.bridge.imgmsg_to_cv2(image, image.encoding)

        if self.system == "modal":
            im_cv = cv2.cvtColor(im_cv, cv2.COLOR_YUV2RGB_Y422)
        else:
            im_cv = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)

        if self.undistort_images:
            im_cv = cv2.undistort(im_cv, self.K, self.dparams, None, None)

        raw_w, raw_h = im_cv.shape[1], im_cv.shape[0]

        if not all([crop == 0 for crop in self.image_crops]):
            im_cv = im_cv[
                self.image_crops[0] : raw_h - self.image_crops[1],
                self.image_crops[2] : raw_w - self.image_crops[3],
            ]

        im_tensor = torch.from_numpy(im_cv).to(dtype=torch.float32) / 255.0
        return True, im_tensor

    def pose_callback(
        self, pose: Union[PoseStamped, Odometry], frame_name: str, is_eval: bool = False
    ) -> Tuple[bool, torch.Tensor]:
        """
        Callback for Pose messages. Extracts pose, converts it to Nerfstudio coordinate
        convention, and inserts it into the Cameras object.
        """
        hom_pose = pose_utils.ros_msg_to_homogenous(pose)

        if self.zero_pose is None:
            if self.center_to_first:
                self.zero_pose = torch.inverse(hom_pose)
            else:
                self.zero_pose = torch.eye(4, dtype=torch.float32)

        hom_pose = self.zero_pose @ hom_pose
        if self.system == "zed":
            c2w = pose_utils.cuvslam_to_nerfstudio(hom_pose)
        else:
            hom_pose = hom_pose @ self.static_transforms[frame_name]
            c2w = pose_utils.opencv_to_nerfstudio(hom_pose)

        # Scale Pose to nerf size
        c2w[:3, 3] *= self.dataset.scene_scale_factor
        device = self.dataset.cameras.device
        c2w = c2w.to(device)

        if is_eval:
            # Don't do radius check if we're just getting the pose for eval
            return True, c2w

        if frame_name in self.last_pose:
            dist = torch.norm(c2w[:3, 3] - self.last_pose[frame_name][:3, 3])
            if dist < self.pose_reject_radius * self.dataset.scene_scale_factor:
                return False, c2w
            else:
                self.last_pose[frame_name] = c2w.clone()
        else:
            self.last_pose[frame_name] = c2w.clone()

        return True, c2w

    def update_dataset_from_buffer(self, step: int):
        """
        Update the dataset's camera poses from the buffer after each training step.
        """
        with torch.no_grad():
            for idx, c2w in self.pose_buffer:
                device = self.dataset.cameras.device
                self.dataset.cameras.camera_to_worlds[idx] = c2w.to(device)
                self.dataset.updated_indices.append(idx)
                self.buffered_idx = idx
        self.pose_buffer.clear()

    def depth_callback(self, depth: Image) -> Tuple[bool, torch.Tensor]:
        """
        Callback for processing Depth Image messages. Similar to RGB image handling,
        but also rescales the depth to the appropriate value.
        """
        depth_cv = self.bridge.imgmsg_to_cv2(depth, depth.encoding)

        raw_w, raw_h = depth_cv.shape[1], depth_cv.shape[0]

        if not all([crop == 0 for crop in self.image_crops]):
            depth_cv = depth_cv[
                self.image_crops[0] : raw_h - self.image_crops[1],
                self.image_crops[2] : raw_w - self.image_crops[3],
            ]

        depth_tensor = torch.from_numpy(depth_cv.astype("float32")).to(
            dtype=torch.float32
        )
        depth_tensor = torch.nan_to_num(depth_tensor)

        depth_tensor *= self.dataset.depth_scale_factor

        depth_tensor[depth_tensor > self.dataset.depth_max] = 0.0
        depth_tensor[depth_tensor < self.dataset.depth_min] = 0.0

        depth_tensor *= self.dataset.scene_scale_factor
        depth_tensor = depth_tensor.unsqueeze(-1)

        return True, depth_tensor

    def pointcloud_callback(self, pc_msg):
        pcd_as_numpy_array = np.array(list(read_points(pc_msg)))
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_as_numpy_array))
        pc_points = np.asarray(pc.points)
        R = self.dataset.pc2c[:3, :3]
        trans = self.dataset.pc2c[:3, 3]
        transformed_points = (R @ pc_points.T).T + trans
        return transformed_points

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def _get_updated_batch(self):
        batch = {}
        for k, v in self.data_dict.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v[: self.current_idx, ...]
        return batch

    def __iter__(self):
        while True:
            if self.updated:
                self.batch = self._get_updated_batch()
                self.updated = False

            batch = self.batch
            yield batch
