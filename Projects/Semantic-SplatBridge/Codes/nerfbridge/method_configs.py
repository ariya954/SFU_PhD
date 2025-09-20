from typing import Union

# Code slightly adapted from Nerfstudio
# https://github.com/nerfstudio-project/nerfstudio/blob/df784e96e7979aaa4320284c087d7036dce67c28/nerfstudio/configs/method_configs.py

"""
NerfBridge Method Configs
"""


from nerfstudio.configs.base_config import (
    ViewerConfig,
    LoggingConfig,
    LocalWriterConfig,
)
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.models.depth_nerfacto import DepthNerfactoModelConfig
from nerfstudio.model_components.losses import DepthLossType
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.pixel_samplers import PairPixelSamplerConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig


from nerfbridge.ros_datamanager import (
    ROSDataManagerConfig,
    ROSFullImageDataManagerConfig,
)
from nerfbridge.ros_pipeline import ROSPipelineConfig
from nerfbridge.ros_dataparser import ROSDataParserConfig
from nerfbridge.ros_trainer import ROSTrainerConfig
from nerfbridge.ros_splatfacto import ROSSplatfactoModelConfig

from nerfbridge.rtguide_pipeline import RTGuidePipelineConfig
from nerfbridge.rtguide_splatfacto import RTGuideModelConfig

RosDepthNerfacto = MethodSpecification(
    config=ROSTrainerConfig(
        method_name="ros-depth-nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=30000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=ROSDataManagerConfig(
                pixel_sampler=PairPixelSamplerConfig(),
                dataparser=ROSDataParserConfig(
                    aabb_scale=1.0,
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=DepthNerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                depth_loss_type=DepthLossType.DS_NERF,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=20000),
        vis="viewer",
    ),
    description="Run NerfBridge with the DepthNerfacto model, and train with streamed RGB and depth images.",
)

RosDepthSplatfacto = MethodSpecification(
    config=ROSTrainerConfig(
        method_name="ros-depth-splatfacto",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=1000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 100},
        pipeline=ROSPipelineConfig(
            datamanager=ROSFullImageDataManagerConfig(
                dataparser=ROSDataParserConfig(aabb_scale=1.0),
            ),
            model=ROSSplatfactoModelConfig(
                num_random=50,
                cull_alpha_thresh=0.25,
                cull_scale_thresh=1.5,
                sh_degree=0,
                use_scale_regularization=True,
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5, max_steps=30000
                ),
            },
            "clip_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=30000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
        logging=LoggingConfig(local_writer=LocalWriterConfig(enable=False)),
    ),
    description="Run NerfBridge with the Splatfacto model, and train with streamed RGB and depth images used to seed new gaussains.",
)

RTGuideSplatfacto = MethodSpecification(
    config=ROSTrainerConfig(
        method_name="rt-guide-splatfacto",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=56000,
        mixed_precision=False,
        pipeline=RTGuidePipelineConfig(
            datamanager=ROSFullImageDataManagerConfig(
                dataparser=ROSDataParserConfig(aabb_scale=1.0),
            ),
            model=RTGuideModelConfig(),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "clip_field": {
                "optimizer": RAdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=30000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
        logging=LoggingConfig(local_writer=LocalWriterConfig(enable=False)),
    ),
    description="Config for RT Guide",
)