"""
This script is used to get the static transforms between multiple cameras
and a base frame. It writes these transforms to a JSON file.

We use this script because the ROS2 tf2_ros library has some threading issues
that cause latency in the rest of the nerfbridge pipeline. This is probably
a configuration issue, but we are using this workaround for now.

Usage:
    1) Set the configuration that you want at the bottom of this file.
    2) Run this script with: python get_tf_transforms.py
    3) Provide a path to the output of this file in the config.json for
        your nerfbridge run.
"""

import rclpy
import rclpy.time
from pathlib import Path
from tf2_ros import Buffer, TransformListener
from nerfbridge import pose_utils
import json
from rich.console import Console


class MultiCameraTFGetter:
    def __init__(self, base_frame_name: str, frame_names: list[str], output_file: Path):
        self.node = rclpy.create_node("multi_camera_tf_getter")
        self.base_frame_name = base_frame_name
        self.frame_names = frame_names
        self.output_file = output_file
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        self.console = Console()
        self.console.print(
            f"Getting transforms for {frame_names} relative to {base_frame_name}"
        )
        self.static_sub = self.node.create_timer(0.1, self.static_pose_callback)

        self.succeeded = False

    def static_pose_callback(self):
        if self.succeeded:
            return

        base_to_camera = {}
        for camera_name in self.frame_names:
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    self.base_frame_name, camera_name, rclpy.time.Time()
                )
            except Exception as e:
                continue

            base_to_camera[camera_name] = pose_utils.get_homog_from_TFMsg(tf_msg)
            base_to_camera[camera_name] = base_to_camera[camera_name].tolist()

        if all([camera_name in base_to_camera for camera_name in self.frame_names]):
            with open(self.output_file, "w") as f:
                json.dump(base_to_camera, f, indent=4)

            self.console.print(f"Successfully saved camera poses to {self.output_file}")
            self.console.print("Use Ctrl+C to exit!")
            self.succeeded = True


if __name__ == "__main__":
    # --------- Configuration ---------
    # Output file destination
    output_path = Path("../configs/static_transforms/three_spot.json")
    # Name of the base frame, this should be the frame that
    # the odometry data uses.
    base_frame_name = "body"
    # List of TF frame names for the cameras.
    frame_names = ["frontleft_fisheye", "left_fisheye", "right_fisheye"]

    # Run the Node
    rclpy.init()
    tf_getter = MultiCameraTFGetter(base_frame_name, frame_names, output_path)
    rclpy.spin(tf_getter.node)
    rclpy.shutdown()
