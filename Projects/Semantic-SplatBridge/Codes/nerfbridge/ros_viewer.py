from typing import Union

from typing import Optional, Dict, Literal, Set

from pathlib import Path
import numpy as np
import trimesh
import rclpy

import viser
import viser.transforms as vtf
from nerfstudio.viewer.viewer import Viewer
from nerfstudio.data.datasets.base_dataset import InputDataset

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from nerfbridge import pose_utils

VISER_NERFSTUDIO_SCALE_RATIO: float = 10.0


class ROSViewer(Viewer):
    """
    A viewer that supports rendering streaming training views.
    """

    def init_scene(
        self,
        train_dataset: InputDataset,
        train_state: Literal["training", "paused", "completed"],
        eval_dataset: Optional[InputDataset] = None,
    ) -> None:
        """Draw some images and the scene aabb in the viewer.

        Args:
            dataset: dataset to render in the scene
            train_state: Current status of training
        """
        # draw the training cameras and images
        self.camera_handles: Dict[int, viser.CameraFrustumHandle] = {}
        self.original_c2w: Dict[int, np.ndarray] = {}
        self.cameras_drawn: Set[int] = set()
        self.dataset = train_dataset
        self.scene_scale = train_dataset.scene_scale_factor

        H, W = self.dataset.image_height, self.dataset.image_width
        W_new = int(100 * W / H)
        image_uint8 = np.ones((100, W_new, 3), dtype=np.uint8)
        image_uint8 = [127, 201, 127] * image_uint8
        eval_image_uint8 = np.ones((100, W_new, 3), dtype=np.uint8)
        eval_image_uint8 = [127, 127, 201] * eval_image_uint8

        if train_state == "training":
            image_indices = self._pick_drawn_image_idxs(len(train_dataset))
            cameras = train_dataset.cameras.to("cpu")
            for idx in image_indices:
                camera = cameras[idx]
                c2w = camera.camera_to_worlds.cpu().numpy()
                R = vtf.SO3.from_matrix(c2w[:3, :3])
                R = R @ vtf.SO3.from_x_radians(np.pi)
                camera_handle = self.viser_server.add_camera_frustum(
                    name=f"/cameras/camera_{idx:05d}",
                    fov=float(2 * np.arctan(camera.cx / camera.fx[0])),
                    scale=self.config.camera_frustum_scale,
                    aspect=float(camera.cx[0] / camera.cy[0]),
                    image=image_uint8,
                    wxyz=R.wxyz,
                    position=c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO,
                    visible=False,
                )

                @camera_handle.on_click
                def _(
                    event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle],
                ) -> None:
                    with event.client.atomic():
                        event.client.camera.position = event.target.position
                        event.client.camera.wxyz = event.target.wxyz

                self.camera_handles[idx] = camera_handle
                self.original_c2w[idx] = c2w
        else:
            c2ws = self.pipeline.data_states["train_c2ws"]
            cameras = train_dataset.cameras.to("cpu")
            for idx in range(len(c2ws)):
                camera = cameras[idx]
                c2w = c2ws[idx].cpu().numpy()
                R = vtf.SO3.from_matrix(c2w[:3, :3])
                R = R @ vtf.SO3.from_x_radians(np.pi)
                camera_handle = self.viser_server.add_camera_frustum(
                    name=f"/cameras/camera_{idx:05d}",
                    fov=float(2 * np.arctan(camera.cx / camera.fx[0])),
                    scale=self.config.camera_frustum_scale,
                    aspect=float(camera.cx[0] / camera.cy[0]),
                    image=image_uint8,
                    wxyz=R.wxyz,
                    position=c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO,
                    visible=True,
                )

                @camera_handle.on_click
                def _(
                    event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle],
                ) -> None:
                    with event.client.atomic():
                        event.client.camera.position = event.target.position
                        event.client.camera.wxyz = event.target.wxyz

                self.camera_handles[idx] = camera_handle
                self.original_c2w[idx] = c2w

            c2ws = self.pipeline.data_states["eval_c2ws"]
            cameras = train_dataset.cameras.to("cpu")
            for idx in range(len(c2ws)):
                camera = cameras[idx]
                c2w = c2ws[idx].cpu().numpy()
                R = vtf.SO3.from_matrix(c2w[:3, :3])
                R = R @ vtf.SO3.from_x_radians(np.pi)
                camera_handle = self.viser_server.add_camera_frustum(
                    name=f"/eval_cameras/camera_{idx:05d}",
                    fov=float(2 * np.arctan(camera.cx / camera.fx[0])),
                    scale=self.config.camera_frustum_scale,
                    aspect=float(camera.cx[0] / camera.cy[0]),
                    image=eval_image_uint8,
                    wxyz=R.wxyz,
                    position=c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO,
                    visible=True,
                )

                @camera_handle.on_click
                def _(
                    event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle],
                ) -> None:
                    with event.client.atomic():
                        event.client.camera.position = event.target.position
                        event.client.camera.wxyz = event.target.wxyz

                self.camera_handles[idx] = camera_handle
                self.original_c2w[idx] = c2w

        self.train_state = train_state
        self.train_util = 0.9
        self.acc_vis_threshold = 0.15

    def robot_pose_callback(self, pose: Union[Odometry, PoseStamped]):
        if self.slam_method == "cuvslam":
            # Odometry Message
            hom_pose = pose_utils.ros_pose_to_homogenous(pose.pose)
            c2w = pose_utils.cuvslam_to_nerfstudio(hom_pose)
        elif self.slam_method == "orbslam3":
            # PoseStamped Message
            hom_pose = pose_utils.ros_pose_to_homogenous(pose)
            c2w = pose_utils.orbslam3_to_nerfstudio(hom_pose)
        elif self.slam_method == "mocap":
            # PoseStamped Message
            hom_pose = pose_utils.ros_pose_to_homogenous(pose)
            c2w = pose_utils.mocap_to_nerfstudio(hom_pose)
        elif self.slam_method == "zedsdk":
            # PoseStamped Message
            hom_pose = pose_utils.ros_pose_to_homogenous(pose)
            c2w = pose_utils.cuvslam_to_nerfstudio(hom_pose)
        elif self.slam_method == "modal":
            # PoseStamped Message
            hom_pose = pose_utils.ros_pose_to_homogenous(pose)
            c2w = pose_utils.opencv_to_nerfstudio(hom_pose)
        else:
            raise NameError("Unknown SLAM Method!")

        if self.scene_scale is not None:
            c2w[:3, 3] *= self.scene_scale

        self.robot_handle.position = (
            c2w[:3, 3].cpu().numpy() * VISER_NERFSTUDIO_SCALE_RATIO
        )

        # Rotate the robot to match the camera orientation (works for drone mesh)
        rot = (
            vtf.SO3.from_matrix(c2w[:3, :3])
            @ vtf.SO3.from_x_radians(-np.pi / 2)
            @ vtf.SO3.from_z_radians(np.pi)
        )
        self.robot_handle.wxyz = rot.wxyz

        # self.robot_pose = c2w

    def setup_robot_viewer_node(self, pose_topic: str, slam_method: str):
        """Returns the viewer node for the scene.

        Args:
            mode: replay or live mode
        Returns:
            The viewer node for the scene
        """
        self.init_robot_mesh()
        self.node = rclpy.create_node("viser_robot_state_node")

        self.slam_method = slam_method
        self.scene_scale = None

        if slam_method == "cuvslam":
            self.node.create_subscription(
                Odometry, pose_topic, self.robot_pose_callback, 10
            )
        elif slam_method == "orbslam3":
            self.node.create_subscription(
                Odometry, pose_topic, self.robot_pose_callback, 10
            )
        elif slam_method == "mocap":
            self.node.create_subscription(
                PoseStamped, pose_topic, self.robot_pose_callback, 10
            )
        elif self.slam_method == "zedsdk":
            self.node.create_subscription(
                PoseStamped, pose_topic, self.robot_pose_callback, 10
            )
        elif self.slam_method == "modal":
            self.node.create_subscription(
                PoseStamped, pose_topic, self.robot_pose_callback, 10
            )
        else:
            raise NameError("Unsupported SLAM algorithm.")

    def init_robot_mesh(self) -> None:
        """Draw the robot mesh in the scene.

        Args:
            robot_mesh: path to the robot mesh
        """
        mesh_path = str(Path(__file__).parent.parent / "meshes/drone.obj")
        mesh = trimesh.load_mesh(mesh_path)
        scale = VISER_NERFSTUDIO_SCALE_RATIO
        self.robot_handle = self.viser_server.add_mesh_trimesh(
            name="/robot",
            mesh=mesh,
            scale=scale,
            wxyz=vtf.SO3.from_x_radians(0.0).wxyz,
            position=(0.0, 0.0, 0.0),
            visible=True,
        )

    def update_camera_poses(self):
        """Updates the camera poses in the scene."""
        image_indices = self.dataset.updated_indices
        for idx in image_indices:
            if (not idx in self.cameras_drawn) and (idx in self.camera_handles):
                self.original_c2w[idx] = (
                    self.dataset.cameras.camera_to_worlds[idx].cpu().numpy()
                )
                self.camera_handles[idx].visible = True
                self.cameras_drawn.add(idx)
        super().update_camera_poses()
