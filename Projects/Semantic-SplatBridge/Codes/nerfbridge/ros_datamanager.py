from typing import Union

"""
A datamanager for the NerfBridge.
"""

from dataclasses import dataclass, field
from typing import (
    Type,
    Dict,
    Tuple,
    Generic,
    Union,
    Literal,
)
from typing_extensions import TypeVar
import random

import torch
import gc
from pathlib import Path

from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.cameras.rays import RayBundle

from nerfbridge.ros_dataset import ROSDataset
from nerfbridge.ros_dataloader import ROSDataloader
from nerfbridge.ros_dataparser import ROSDataParserConfig
from nerfbridge.writer import CONSOLE


@dataclass
class ROSDataManagerConfig(VanillaDataManagerConfig):
    """A ROS datamanager that handles a streaming dataloader."""

    _target: Type = field(default_factory=lambda: ROSDataManager)
    dataparser: ROSDataParserConfig = ROSDataParserConfig()
    """ Must use only the ROSDataParser here """
    num_training_images: int = 600
    """ Number of images to train on (for dataset tensor pre-allocation). """
    use_semantics: bool = True
    """ Whether to use semantic features or not."""


TDataset = TypeVar("TDataset", bound=ROSDataset, default=ROSDataset)


class ROSDataManager(VanillaDataManager, Generic[TDataset]):  # pylint: disable=abstract-method
    """Essentially the VannilaDataManager from Nerfstudio except that the
    typical dataloader for training images is replaced with one that streams
    image and pose data from ROS.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: ROSDataManagerConfig
    train_dataset: ROSDataset

    def __init__(
        self,
        config: VanillaDataManagerConfig,
        cache_dir: Path,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        run_mode: Literal["train", "eval"] = "train",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        # Set the cache directory
        self.cache_dir = cache_dir
        self.run_mode = run_mode

        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            **kwargs,
        )

    def create_train_dataset(self) -> ROSDataset:
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(
            split="train", num_images=self.config.num_training_images
        )
        return ROSDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.train_dataparser_outputs.dataparser_scale,
            device=self.device,
        )

    def setup_train(self):
        assert self.train_dataset is not None
        self.train_image_dataloader = ROSDataloader(
            self.train_dataset,
            self.config.use_semantics,
            device=self.device,
            num_workers=0,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            cache_dir=self.cache_dir,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(
            self.train_dataset, self.config.train_num_rays_per_batch
        )
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras)

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """
        First, checks for updates to the ROSDataloader, and then returns the next
        batch of data from the train dataloader.
        """
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)

        return ray_bundle, batch

    def setup_eval(self):
        """
        Evaluation data is not implemented! This function is called by
        the parent class, but the results are never used.
        """
        pass

    def create_eval_dataset(self):
        """
        Evaluation data is not implemented! This function is called by
        the parent class, but the results are never used.
        """
        pass

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        CONSOLE.print("Evaluation data is not setup!")
        raise NameError(
            "Evaluation funcationality not yet implemented with ROS Streaming."
        )

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        CONSOLE.print("Evaluation data is not setup!")
        raise NameError(
            "Evaluation funcationality not yet implemented with ROS Streaming."
        )


@dataclass
class ROSFullImageDataManagerConfig(ROSDataManagerConfig):
    """
    A ROS DataManager that serves full images (for rasterization) from
    a streaming dataloader. Configuration is all the same as the ROSDataManager.
    """

    _target: Type = field(default_factory=lambda: ROSFullImageDataManager)


class ROSFullImageDataManager(ROSDataManager, Generic[TDataset]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize features
        self.train_features = {}

        # Garbage collect
        torch.cuda.empty_cache()
        gc.collect()

    def setup_train(self):
        """
        Sets up the dataloader for serving full images from ROS.
        """
        assert self.train_dataset is not None
        self.train_image_dataloader = ROSDataloader(
            self.train_dataset,
            self.config.use_semantics,
            self.cache_dir,
            device=self.device,
            num_workers=0,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.latest_image_idx = 0
        self.unseen_images = []
        self.recent_images = []

    def get_train_rays_per_batch(self) -> int:
        H = self.train_dataset.image_height
        W = self.train_dataset.image_width
        return H * W

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """
        First, checks for updates to the ROSDataloader, and then returns the next
        batch of data from the train dataloader.

        Images are returned with the following priority:
            1. Recently recieved images that have not been seen before
            2. Images not seen from the last loop through the dataset up
                to the latest image index.
        """
        if len(self.unseen_images) == 0:
            self.unseen_images = [i for i in range(self.latest_image_idx + 1)]

        if self.latest_image_idx < self.train_image_dataloader.buffered_idx:
            # New images have been recieved
            prev_latest = self.latest_image_idx
            self.latest_image_idx = self.train_image_dataloader.buffered_idx
            self.recent_images.extend(range(prev_latest, self.latest_image_idx + 1))

        if len(self.recent_images) > 0:
            idx = self.recent_images.pop(0)
        else:
            idx = self.unseen_images.pop(random.randint(0, len(self.unseen_images) - 1))

        self.train_count += 1

        image_data = self.train_image_dataloader[idx]
        image_data["image"] = image_data["image"].to(self.device)
        if "depth_image" in image_data:
            image_data["depth_image"] = image_data["depth_image"].to(self.device)

        # get the semantic features
        if self.config.use_semantics:
            image_data["clip"] = self.train_dataset.semantics_tensor[idx].squeeze()
            image_data["clip"] = image_data["clip"].to(self.device)

        camera = self.train_dataset.cameras[idx : idx + 1].to(self.device)

        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = idx

        return camera, image_data
