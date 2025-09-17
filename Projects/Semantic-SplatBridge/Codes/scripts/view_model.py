#!/usr/bin/env python
from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from threading import Lock
from typing import Literal
import os

import tyro

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import writer

from nerfbridge.eval import load_from_checkpoint
from nerfbridge.ros_viewer import ROSViewer


@dataclass
class ViewerConfigWithoutNumRays(ViewerConfig):
    """Configuration for viewer instantiation"""

    num_rays_per_chunk: int = -1

    def as_viewer_config(self):
        """Converts the instance to ViewerConfig"""
        return ViewerConfig(**{x.name: getattr(self, x.name) for x in fields(self)})


@dataclass
class RunViewer:
    """Load a checkpoint and start the viewer."""

    load_config: Path
    """Path to config YAML file."""
    viewer: ViewerConfigWithoutNumRays = field(default_factory=ViewerConfigWithoutNumRays)
    """Viewer configuration"""
    vis: Literal["viewer", "viewer_legacy"] = "viewer"
    """Type of viewer"""

    def main(self) -> None:
        """Main function."""
        # config, pipeline, _, step = eval_setup(
        #     self.load_config,
        #     eval_num_rays_per_chunk=None,
        #     test_mode="test",
        # )
        # Path to checkpoint file
        checkpoint_dir = self.load_config.parent / "nerfstudio_models"
        load_step = sorted(
            int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(checkpoint_dir)
        )[-1]
        latest_checkpoint = checkpoint_dir / f"step-{load_step:09d}.ckpt"

        config, pipeline = load_from_checkpoint(self.load_config, latest_checkpoint)

        num_rays_per_chunk = config.viewer.num_rays_per_chunk
        assert self.viewer.num_rays_per_chunk == -1
        config.vis = self.vis
        config.viewer = self.viewer.as_viewer_config()
        config.viewer.num_rays_per_chunk = num_rays_per_chunk

        _start_viewer(config, pipeline, load_step)

    def save_checkpoint(self, *args, **kwargs):
        """
        Mock method because we pass this instance to viewer_state.update_scene
        """


def _start_viewer(config: TrainerConfig, pipeline: Pipeline, step: int):
    """Starts the viewer

    Args:
        config: Configuration of pipeline to load
        pipeline: Pipeline instance of which to load weights
        step: Step at which the pipeline was saved
    """
    base_dir = config.get_base_dir()
    viewer_log_path = base_dir / config.viewer.relative_log_filename
    banner_messages = None
    viewer_state = None
    viewer_callback_lock = Lock()

    viewer_state = ROSViewer(
        config.viewer,
        log_filename=viewer_log_path,
        datapath=pipeline.datamanager.get_datapath(),
        pipeline=pipeline,
        share=config.viewer.make_share_url,
        train_lock=viewer_callback_lock,
    )
    banner_messages = viewer_state.viewer_info

    # We don't need logging, but writer.GLOBAL_BUFFER needs to be populated
    config.logging.local_writer.enable = False
    writer.setup_local_writer(config.logging, max_iter=config.max_num_iterations, banner_messages=banner_messages)

    assert viewer_state and pipeline.datamanager.train_dataset
    viewer_state.init_scene(
        train_dataset=pipeline.datamanager.train_dataset,
        train_state="completed",
        eval_dataset=pipeline.datamanager.eval_dataset,
    )
    viewer_state.update_scene(step=step)
    while True:
        time.sleep(0.01)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[RunViewer]).main()


if __name__ == "__main__":
    entrypoint()
