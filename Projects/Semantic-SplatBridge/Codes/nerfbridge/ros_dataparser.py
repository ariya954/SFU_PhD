from typing import Union

"""Data parser for loading ROS parameters."""


from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json

from nerfbridge.writer import CONSOLE


@dataclass
class ROSDataParserConfig(DataParserConfig):
    """ROS config file parser config."""

    _target: Type = field(default_factory=lambda: ROSDataParser)
    """target class to instantiate"""
    data: Path = Path("data/ros/nerfbridge_config.json")
    """ Path to configuration JSON. """
    aabb_scale: float = 1.0
    """ SceneBox aabb scene side L = [-scale, scale]"""


@dataclass
class ROSDataParser(DataParser):
    """ROS DataParser"""

    config: ROSDataParserConfig

    def __init__(self, config: ROSDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.aabb: float = config.aabb_scale

    def get_dataparser_outputs(self, split="train", num_images: int = 500):
        dataparser_outputs = self._generate_dataparser_outputs(split, num_images)
        return dataparser_outputs

    def _generate_dataparser_outputs(self, split="train", num_images: int = 500):
        """
        This function generates a DataParserOutputs object. Typically in Nerfstudio
        this is used to populate the training and evaluation datasets, but since with
        NerfBridge our aim is to stream the data then we only have to worry about
        loading the proper camera parameters and ROS topic names.

        Args:
            split: Determines the data split (not used, but left in place for consistency
                with Nerfstudio)

            num_images: The size limit of the training image dataset. This is used to
                pre-allocate tensors for the Cameras object that tracks camera pose.
        """
        meta = load_from_json(self.data)

        image_height = meta["H"]
        image_width = meta["W"]
        fx = meta["fx"]
        fy = meta["fy"]
        cx = meta["cx"]
        cy = meta["cy"]

        # Cropping images
        # top, bottom, left, right
        image_crops = meta["image_crops"]
        assert len(image_crops) == 4
        top_crop, bottom_crop, left_crop, right_crop = image_crops

        if top_crop > 0:
            assert top_crop < image_height
            image_height -= top_crop
            cy -= top_crop

        if bottom_crop > 0:
            assert bottom_crop < image_height
            image_height -= bottom_crop

        if left_crop > 0:
            assert left_crop < image_width
            image_width -= left_crop
            cx -= left_crop

        if right_crop > 0:
            assert right_crop < image_width
            image_width -= right_crop

        CONSOLE.print(
            f"Image size: {image_width} x {image_height} (cropped from {meta['W']} x {meta['H']})",
            style="yellow",
        )

        # Distortion parameters
        k1 = meta["k1"] if "k1" in meta else 0.0
        k2 = meta["k2"] if "k2" in meta else 0.0
        k3 = meta["k3"] if "k3" in meta else 0.0
        k4 = meta["k4"] if "k4" in meta else 0.0
        p1 = meta["p1"] if "p1" in meta else 0.0
        p2 = meta["p2"] if "p2" in meta else 0.0
        distort = torch.tensor([k1, k2, k3, k4, p1, p2], dtype=torch.float32)
        camera_to_world = torch.stack(num_images * [torch.eye(4, dtype=torch.float32)])[
            :, :-1, :
        ]

        # in x,y,z order
        scene_size = self.aabb
        scene_box = SceneBox(
            aabb=torch.tensor(
                [
                    [-scene_size, -scene_size, -scene_size],
                    [scene_size, scene_size, scene_size],
                ],
                dtype=torch.float32,
            )
        )

        # Create a dummy Cameras object with the appropriate number
        # of placeholders for poses.
        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=image_height,
            width=image_width,
            distortion_params=distort,
            camera_type=CameraType.PERSPECTIVE,
        )

        # Create a new dictionary with the correct keys
        image_filenames = []
        metadata = {
            "system": meta["system"],
            "image_topic": meta["image_topic"],
            "pose_topic": meta["pose_topic"],
            "topic_sync": meta["topic_sync"],
            "compressed_rgb": meta["compressed_rgb"],
            "undistort_rgb": meta["undistort_rgb"],
            "pose_reject_radius": meta["pose_reject_radius"],
            "center_to_first_pose": meta["center_to_first_pose"],
            "data_update_freq": meta["data_update_freq"],
            "eval_save_freq": meta["eval_save_freq"],
            "num_images": num_images,
            "image_height": image_height,
            "image_width": image_width,
            "semantics_positives": meta["semantics_positives"],
            "semantics_negatives": meta["semantics_negatives"],
            "image_crops": image_crops,
        }

        # Only used if depth training is enabled
        if meta["topic_sync"] == "approximate":
            metadata["topic_slop"] = meta["topic_slop"]
        else:
            assert meta["topic_sync"] == "exact"
            metadata["topic_slop"] = None

        metadata["depth_topic"] = meta["depth_topic"]
        metadata["depth_scale_factor"] = meta["depth_scale_factor"]
        metadata["depth_min"] = meta["depth_min"]
        metadata["depth_max"] = meta["depth_max"]

        metadata["scene_scale_factor"] = meta["scene_scale_factor"]

        stf_path = Path(meta["static_transforms_file"])
        stf_path = self.data.parent / stf_path
        if stf_path.exists():
            static_transforms_lists = load_from_json(stf_path)
            static_transforms = {
                key: torch.tensor(val).to(dtype=torch.float32)
                for key, val in static_transforms_lists.items()
            }
            metadata["camera_frame_names"] = meta["camera_frame_names"]
            metadata["static_transforms"] = static_transforms
            if meta["system"] == "spot":
                assert all([key in static_transforms for key in metadata["camera_frame_names"]])
        else:
            CONSOLE.print("(IGNORE IF EVALUATION) Static transforms file not found.", style="red")

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,  # This is empty
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata,
        )

        return dataparser_outputs
