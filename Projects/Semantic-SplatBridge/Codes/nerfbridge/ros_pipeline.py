from typing import Union

from dataclasses import dataclass, field
from typing import Literal, Type, Optional, Dict, Any
from pathlib import Path

import torch
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)

from nerfbridge.ros_datamanager import (
    ROSFullImageDataManagerConfig,
)
from nerfbridge.ros_splatfacto import ROSSplatfactoModelConfig


@dataclass
class ROSPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: ROSPipeline)
    """target class to instantiate"""
    datamanager: ROSFullImageDataManagerConfig = ROSFullImageDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = ROSSplatfactoModelConfig()
    """specifies the model config"""


class ROSPipeline(VanillaPipeline):
    def __init__(
        self,
        config: ROSPipelineConfig,
        device: str,
        cache_dir: Path,
        test_mode: Literal["test", "val", "inference"] = "val",
        run_mode: Literal["train", "eval"] = "train",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.run_mode = run_mode

        self.datamanager: ROSFullImageDataManagerConfig = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            cache_dir=cache_dir,
            run_mode=run_mode,
        )

        if run_mode == "eval" and config.datamanager.use_semantics:
            # Load a cached semantics tensor to get the feature dim
            semantics_path = list((cache_dir / "train" / "semantics").glob("*.pt"))[0]
            semantics_tensor = torch.load(semantics_path)
            self.datamanager.train_dataset.metadata["feature_dim"] = (
                semantics_tensor.shape[-1]
            )

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            datamanager=self.datamanager,
            grad_scaler=grad_scaler,
            seed_points=None,
        )
        self.model.to(device)

    def load_pipeline(
        self, loaded_state: Dict[str, Any], data_states: Dict[str, Any], step: int
    ) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        super().load_pipeline(loaded_state, step)
        self.data_states = data_states
