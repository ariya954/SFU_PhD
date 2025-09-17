from typing import Union


import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset


class ROSDataset(InputDataset):
    """
    This is a tensor dataset that keeps track of all of the data streamed by ROS.
    It's main purpose is to conform to the already defined workflow of nerfstudio:
        (dataparser -> inputdataset -> dataloader).

    In reality we could just store everything directly in ROSDataloader, but this
    would require rewritting more code than its worth.

    Images are tracked in self.image_tensor with uninitialized images set to
    all white (hence torch.ones).
    Poses are stored in self.cameras.camera_to_worlds as 3x4 transformation tensors.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(
        self,
        dataparser_outputs: DataparserOutputs,
        scale_factor: float = 1.0,
        device: Union[torch.device, str] = "cpu",
    ):
        super().__init__(dataparser_outputs, scale_factor)
        assert (
            "image_topic" in dataparser_outputs.metadata.keys()
            and "pose_topic" in dataparser_outputs.metadata.keys()
            and "num_images" in dataparser_outputs.metadata.keys()
        )
        self.image_topic_name = self.metadata["image_topic"]
        self.pose_topic_name = self.metadata["pose_topic"]
        self.num_images = self.metadata["num_images"]
        assert self.num_images > 0
        self.image_height = self.metadata["image_height"]
        self.image_width = self.metadata["image_width"]
        self.scene_scale_factor = self.metadata["scene_scale_factor"]
        self.device = device

        # If using spot then use depth images, else use pointclouds
        self.system = self.metadata["system"]
        self.use_depth = False if self.system == "modal" else True

        # Load modal specific configs
        if self.system == "modal":
            self.pc2c = self.metadata["static_transforms"]["pc_to_camera"]
            self.pose2c = self.metadata["static_transforms"]["pose_to_camera"]

        self.cameras = self.cameras.to(device=self.device)

        self.image_tensor = torch.ones(
            self.num_images, self.image_height, self.image_width, 3, dtype=torch.float32
        )
        self.image_indices = torch.arange(self.num_images)

        assert "depth_topic" in dataparser_outputs.metadata.keys()
        assert "depth_scale_factor" in dataparser_outputs.metadata.keys()

        self.depth_topic_name = self.metadata["depth_topic"]
        self.depth_scale_factor = self.metadata["depth_scale_factor"]
        if self.use_depth:
            self.depth_tensor = torch.ones(
                self.num_images,
                self.image_height,
                self.image_width,
                1,
                dtype=torch.float32,
            )
        else:
            self.pc_dict = {}
        self.depth_min = self.metadata["depth_min"]
        self.depth_max = self.metadata["depth_max"]

        # Used for undistortion in the dataloader
        self.K = self.cameras.get_intrinsics_matrices()[0].cpu()
        dps = self.cameras.distortion_params[0]
        self.dparams = torch.tensor([dps[0], dps[1], dps[2], dps[4], dps[5]]).cpu()

        self.updated_indices = []

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx: int):
        """
        This returns the data as a dictionary which is not actually how it is
        accessed in the dataloader, but we allow this as well so that we do not
        have to rewrite the several downstream functions.
        """
        data = {
            "image_idx": idx,
            "image": self.image_tensor[idx],
        }
        if self.use_depth:
            data["depth_image"] = self.depth_tensor[idx]
        return data
