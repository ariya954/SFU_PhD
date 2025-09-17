#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import json

class ImageDepthSaver(Node):
    def __init__(self):
        super().__init__('image_depth_saver')
        self.bridge = CvBridge()
        self.rgb_count = 0
        self.depth_count = 0
        self.rgb_dir = os.path.expanduser('~/Semantic-SplatBridge/data/images')
        self.depth_dir = os.path.expanduser('~/Semantic-SplatBridge/data/depth')
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        self.frames = []

        # RGB subscription
        self.create_subscription(
            Image,
            '/zed/zed_node/rgb/image_rect_color/compressed',
            self.rgb_callback,
            10
        )

        # Depth subscription
        self.create_subscription(
            Image,
            '/zed/zed_node/depth/depth_registered',
            self.depth_callback,
            10
        )

    def rgb_callback(self, msg):
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        rgb_file = os.path.join(self.rgb_dir, f'{self.rgb_count:06d}.png')
        cv2.imwrite(rgb_file, cv_image)
        self.rgb_count += 1
        self.get_logger().info(f'Saved RGB {rgb_file}')

    def depth_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        depth_file = os.path.join(self.depth_dir, f'{self.depth_count:06d}.png')
        cv2.imwrite(depth_file, cv_image)
        self.depth_count += 1
        self.get_logger().info(f'Saved Depth {depth_file}')

def main(args=None):
    rclpy.init(args=args)
    saver = ImageDepthSaver()
    try:
        rclpy.spin(saver)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
