from typing import Union

from dataclasses import dataclass, field
from typing import Type, List, Dict, Tuple, Union

import cv2

import torch
import numpy as np

from torch.nn import Parameter

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

from nerfstudio.models.splatfacto import (
    SplatfactoModel,
    SplatfactoModelConfig,
    RGB2SH,
    num_sh_bases,
    random_quat_tensor,
    get_viewmat,
)
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.colormaps import ColormapOptions, apply_colormap
from nerfbridge.viewer_utils import ViewerUtils
from nerfstudio.viewer.server.viewer_elements import (
    ViewerButton,
    ViewerNumber,
    ViewerText,
)

try:
    import tinycudann as tcnn
except ImportError:
    pass


@dataclass
class RTGuideModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: RTGuideModel)
    depth_seed_pts: int = 1000
    """ Number of points to use for seeding the model from depth per image. """
    seed_every: int = 1
    """ Seed every k keyframes. """
    """ Pointcloud maximum seed distance. """
    output_depth_during_training: bool = True
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    enable_depth_loss: bool = False
    """Option to enable depth loss."""
    depth_loss_mult: float = 1e-1
    """weight for the depth-related term in the loss function."""

    # Semantic field
    enable_semantics: bool = True
    """Option to enable semantic field."""
    enable_clip_mask: bool = False
    """Option to utilize the alpha channel as a mask for CLIP distillation."""
    semantics_batch_size: int = 1
    """The batch size for training the semantic field."""
    output_semantics_during_training: bool = True
    """If True, output semantic-scene information during training. Otherwise, only output semantic-scene information during evaluation."""
    clip_img_loss_weight: float = 1e0
    """weight for the CLIP-related term in the loss function."""
    semantic_field_scale: float = 0.3
    """Scale factor for the semantic field."""

    # MLP head
    hidden_dim: int = 64
    num_layers: int = 2

    # Positional encoding
    use_pe: bool = True
    pe_n_freq: int = 6
    # Hash grid
    num_levels: int = 12
    log2_hashmap_size: int = 19
    start_res: int = 16
    max_res: int = 128
    features_per_level: int = 8
    hashgrid_layers: Tuple[int, ...] = (12, 12)
    hashgrid_resolutions: Tuple[Tuple[int, ...]] = ((16, 128), (128, 512))
    hashgrid_sizes: Tuple[int, ...] = (19, 19)


class RTGuideModel(SplatfactoModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seeded_img_idx = 0
        self.depth_seed_pts = self.config.depth_seed_pts

        self.alpha = 5
        self.beta = 5

        # For some reason this is not set in the base class
        self.vis_counts = None

        self.seed_every = self.config.seed_every

    def populate_modules(self):
        super().populate_modules()
        # Move the initial means far away (THIS IS A HACK AND SHOULD BE FIXED)
        self.gauss_params["means"] = self.gauss_params["means"] + 100.0
        self.param_delta = None

        if self.config.enable_semantics:
            # CLIP embeddings input dimension
            self.clip_embeds_input_dim = self.kwargs["metadata"]["feature_dim"]

            # Feature field has its own hash grid
            growth_factor = np.exp(
                (np.log(self.config.max_res) - np.log(self.config.start_res))
                / (self.config.num_levels - 1)
            )
            encoding_config = {
                "otype": "Composite",
                "nested": [
                    {
                        "otype": "HashGrid",
                        "n_levels": self.config.num_levels,
                        "n_features_per_level": self.config.features_per_level,
                        "log2_hashmap_size": self.config.log2_hashmap_size,
                        "base_resolution": self.config.start_res,
                        "per_level_scale": growth_factor,
                    }
                ],
            }

            if self.config.use_pe:
                encoding_config["nested"].append(
                    {
                        "otype": "Frequency",
                        "n_frequencies": self.config.pe_n_freq,
                        "n_dims_to_encode": 3,
                    }
                )

            self.clip_field = tcnn.NetworkWithInputEncoding(
                n_input_dims=3,
                n_output_dims=self.clip_embeds_input_dim,
                encoding_config=encoding_config,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.config.hidden_dim,
                    "n_hidden_layers": self.config.num_layers,
                },
            )

            # initialize Viewer
            self.viewer_utils = ViewerUtils()

            self.setup_gui()

    def setup_gui(self):
        self.viewer_utils.device = "cuda:0"  # self.device #"cuda:0"
        # Note: the GUI elements are shown based on alphabetical variable names
        self.btn_refresh_pca = ViewerButton(
            "Refresh PCA Projection",
            cb_hook=lambda _: self.viewer_utils.reset_pca_proj(),
        )

        # Only setup GUI for language features if we're using CLIP
        self.hint_text = ViewerText(
            name="Note:", disabled=True, default_value="Use , to separate labels"
        )
        self.lang_1_pos_text = ViewerText(
            name="Language (Positives)",
            default_value="",
            cb_hook=lambda elem: self.viewer_utils.handle_language_queries(
                elem.value, is_positive=True
            ),
        )
        self.lang_2_neg_text = ViewerText(
            name="Language (Negatives)",
            default_value="",
            cb_hook=lambda elem: self.viewer_utils.handle_language_queries(
                elem.value, is_positive=False
            ),
        )
        self.softmax_temp = ViewerNumber(
            name="Softmax temperature",
            default_value=self.viewer_utils.softmax_temp,
            cb_hook=lambda elem: self.viewer_utils.update_softmax_temp(elem.value),
        )

    def seed_cb(self, pipeline: Pipeline, optimizers: Optimizers, step: int):
        ds_latest_idx = pipeline.datamanager.train_image_dataloader.buffered_idx

        if self.seeded_img_idx < ds_latest_idx:
            start_idx = 0 if self.seeded_img_idx == 0 else self.seeded_img_idx + 1
            seed_image_idxs = range(start_idx, ds_latest_idx + 1)
            pre_gaussian_count = self.means.shape[0]
            for idx in seed_image_idxs:
                if idx % self.seed_every != 0:
                    # Only seed every k keyframes
                    continue
                image_data = pipeline.datamanager.train_dataset[idx]
                camera = pipeline.datamanager.train_dataset.cameras[idx]
                with torch.no_grad():
                    if pipeline.datamanager.train_dataset.use_depth:
                        self.seed_from_rgbd(camera, image_data, optimizers)
                    else:
                        pc = pipeline.datamanager.train_dataset.pc_dict[idx]
                        self.seed_from_pc(
                            camera,
                            pc,
                            optimizers,
                            min_distance=pipeline.datamanager.train_dataset.depth_min,
                            max_distance=pipeline.datamanager.train_dataset.depth_max,
                        )
                self.seeded_img_idx = idx
            post_gaussian_count = self.means.shape[0]
            diff_gaussians = post_gaussian_count - pre_gaussian_count

            if diff_gaussians == 0:
                return

            if self.xys_grad_norm is not None:
                device = self.xys_grad_norm.device
                self.xys_grad_norm = torch.cat(
                    [self.xys_grad_norm, torch.zeros(diff_gaussians).to(device)]
                )
            if self.max_2Dsize is not None:
                device = self.max_2Dsize.device
                self.max_2Dsize = torch.cat(
                    [self.max_2Dsize, torch.zeros(diff_gaussians).to(device)]
                )
            if self.vis_counts is not None:
                device = self.vis_counts.device
                self.vis_counts = torch.cat(
                    [self.vis_counts, torch.zeros(diff_gaussians).to(device)]
                )

    def seed_from_rgbd(
        self,
        camera: Cameras,
        image_data: Dict[str, torch.Tensor],
        optimizers: Optimizers,
    ):
        """
        Initialize gaussians at random points in the point cloud from the depth image.

        Means - Initialized to projected points in the depth image.
        Scales - Initialized using k-nearest neighbors approach (same as splatfacto).
        Quats - Initialized to random (same as splatfacto).
        Opacities - Initialized to logit(0.1) (same as splatfacto).
        Features_SH - Initialized to RGB2SH of the points color.
        """
        depth = image_data["depth_image"]
        rgb = image_data["image"]
        H, W, _ = image_data["depth_image"].shape
        depth = torch.nan_to_num(depth, nan=0.0)
        if rgb.device != self.device or depth.device != self.device:
            depth = depth.to(self.device)
            rgb = rgb.to(self.device)

        # Get camera intrinsics and extrinsics
        assert len(camera.shape) == 0
        assert H == camera.image_height.item() and W == camera.image_width.item()
        fx, fy = camera.fx[0].item(), camera.fy[0].item()
        cx, cy = camera.cx[0].item(), camera.cy[0].item()
        c2w = camera.camera_to_worlds.to(self.device)  # (3, 4)
        R = c2w[:3, :3]
        t = c2w[:3, 3].squeeze()

        # Sample pixel indices
        # Could use a confidence map here if available
        nz_row, nz_col = torch.where(depth.squeeze() > 0)
        num_samples = min(self.depth_seed_pts, nz_row.shape[0])
        ind_mask = torch.randperm(nz_row.shape[0])[:num_samples]
        x = nz_col[ind_mask].to(self.device).reshape((-1, 1))
        y = nz_row[ind_mask].to(self.device).reshape((-1, 1))
        rgbs = rgb[y, x, :]  # (num_seed_points, 3)
        rgbs = rgbs.squeeze()

        # Sample depth pixels and project to 3D coordinates (camera relative).
        z = depth[y, x]
        z = z.reshape((-1, 1))  # (num_seed_points, 1)
        x = (x - cx) * z / fx
        y = (y - cy) * z / fy

        # Flip y and z to switch to opengl coordinate system.
        xyzs = torch.stack([x, -y, -z], dim=-1).squeeze()  # (num_seed_points, 3)

        # Transform camera relative 3D coordinates to world coordinates.
        xyzs = torch.matmul(xyzs, R.T) + t  # (num_seed_points, 3)

        # Initialize scales to one pixel spherical
        distances = torch.norm(xyzs, dim=1)
        scales = distances.unsqueeze(1).repeat(1, 3).to(self.device)
        fx = camera.fx[0].item()
        scales = torch.log(scales / fx)

        # Initialize quats to random.
        quats = random_quat_tensor(self.depth_seed_pts).to(self.device)

        # Initialize SH features to RGB2SH of the points color.
        dim_sh = num_sh_bases(self.config.sh_degree)
        shs = torch.zeros((self.depth_seed_pts, dim_sh, 3)).float().to(self.device)
        if self.config.sh_degree > 0:
            shs[:, 0, :3] = RGB2SH(rgbs)
            shs[:, 1:, 3:] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(rgbs, eps=1e-10)
        features_dc = shs[:, 0, :]
        features_rest = shs[:, 1:, :]

        opacities = torch.logit(0.3 * torch.ones(self.depth_seed_pts, 1)).to(
            self.device
        )

        # Concatenate the new gaussians to the existing ones.
        self.gauss_params["means"] = torch.nn.Parameter(
            torch.cat([self.means.detach(), xyzs], dim=0)
        )
        self.gauss_params["scales"] = torch.nn.Parameter(
            torch.cat([self.scales.detach(), scales], dim=0)
        )
        self.gauss_params["quats"] = torch.nn.Parameter(
            torch.cat([self.quats.detach(), quats], dim=0)
        )
        self.gauss_params["opacities"] = torch.nn.Parameter(
            torch.cat([self.opacities.detach(), opacities], dim=0)
        )
        self.gauss_params["features_dc"] = torch.nn.Parameter(
            torch.cat([self.features_dc.detach(), features_dc], dim=0)
        )
        self.gauss_params["features_rest"] = torch.nn.Parameter(
            torch.cat([self.features_rest.detach(), features_rest], dim=0)
        )

        # Add the new parameters to the optimizer.
        for param_group, new_param in self.get_gaussian_param_groups().items():
            optimizer = optimizers.optimizers[param_group]
            old_param = optimizer.param_groups[0]["params"][0]
            param_state = optimizer.state[old_param]
            added_param_shape = (self.depth_seed_pts, *new_param[0].shape[1:])
            if "exp_avg" in param_state:
                param_state["exp_avg"] = torch.cat(
                    [
                        param_state["exp_avg"],
                        torch.zeros(added_param_shape).to(self.device),
                    ],
                    dim=0,
                )
            if "exp_avg_sq" in param_state:
                param_state["exp_avg_sq"] = torch.cat(
                    [
                        param_state["exp_avg_sq"],
                        torch.zeros(added_param_shape).to(self.device),
                    ],
                    dim=0,
                )

            del optimizer.state[old_param]
            optimizer.state[new_param[0]] = param_state
            optimizer.param_groups[0]["params"] = new_param
            del old_param

    def seed_from_pc(
        self,
        camera: Cameras,
        pc_data,
        optimizers: Optimizers,
        min_distance: float = 0.1,
        max_distance: float = 5.0,
    ):
        """
        Initialize gaussians at random points in the point cloud from the depth image.

        Means - Initialized to projected points in the depth image.
        Scales - Initialized using k-nearest neighbors approach (same as splatfacto).
        Quats - Initialized to random (same as splatfacto).
        Opacities - Initialized to logit(0.1) (same as splatfacto).
        Features_SH - Initialized to RGB2SH of the points color.
        """

        fx, fy = camera.fx[0].item(), camera.fy[0].item()
        cx, cy = camera.cx[0].item(), camera.cy[0].item()
        H = camera.image_height[0].item()
        W = camera.image_width[0].item()
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        c2w = camera.camera_to_worlds.to(self.device)  # (3, 4)
        R = c2w[:3, :3]
        t = c2w[:3, 3].squeeze()

        pc_data = pc_data[~torch.all(pc_data == 0, axis=1)]

        # Project the point cloud to the image plane
        # Remove points not in RGB FOV
        proj, _ = cv2.projectPoints(
            pc_data.numpy(), np.zeros(3), np.zeros(3), K, np.zeros(5)
        )
        proj = torch.from_numpy(proj.squeeze())
        umask = (0 <= proj[:, 0]) & (proj[:, 0] < W)
        vmask = (0 <= proj[:, 1]) & (proj[:, 1] < H)
        in_view = umask & vmask
        pc_data = pc_data[in_view]

        xyzs = pc_data.to(dtype=torch.float32)  # current size (16480, 3) torch tensor

        # Assuming x, y, and z are the individual components from xyzs
        x = xyzs[:, 0]
        y = xyzs[:, 1]
        z = xyzs[:, 2]

        # Flip y and z by negating them and stack them back together
        xyzs = torch.stack([x, -y, -z], dim=-1).squeeze()  # (num_seed_points, 3)

        # Calculate the Euclidean distance to the origin for each point
        distances = torch.norm(xyzs, dim=1)

        # Create a mask to filter points that are within the allowable distance range
        mask = (distances > min_distance) & (distances < max_distance)
        n_points = mask.sum()
        if n_points == 0:
            return
        elif n_points < self.depth_seed_pts:
            local_n_seed = n_points
        else:
            local_n_seed = self.depth_seed_pts

        # Apply the mask to the point cloud
        xyzs = xyzs[mask]

        # Calculate the range (min, max) of the distances
        indices = torch.randperm(xyzs.size(0))[:local_n_seed]

        # Use the indices to select the downsampled point cloud
        xyzs = xyzs[indices]

        xyzs = xyzs.to(self.device)
        xyzs = torch.matmul(xyzs, R.T) + t  # (num_seed_points, 3)

        # Initialize scales using 3-nearest neighbors average distance.
        ## Spawn them all as the same size gaussian for now
        distances = torch.norm(xyzs, dim=1)
        scales = distances.unsqueeze(1).repeat(1, 3).to(self.device)
        fx = camera.fx[0].item()
        scales = torch.log(scales / fx)

        # Initialize quats to random.
        quats = random_quat_tensor(local_n_seed).to(self.device)

        # Initialize SH features to RGB2SH of the points color.
        dim_sh = num_sh_bases(self.config.sh_degree)
        # lets make the shs init to random
        shs = torch.rand((local_n_seed, dim_sh, 3)).float().to(self.device)
        features_dc = shs[:, 0, :]
        features_rest = shs[:, 1:, :]

        # Initialize opacities to logit(0.3). This is sort of our opacity prior.
        # Nerfstudio uses a opacity prior of 0.1.
        opacities = torch.logit(0.1 * torch.ones(local_n_seed, 1)).to(self.device)

        # Concatenate the new gaussians to the existing ones.
        self.gauss_params["means"] = torch.nn.Parameter(
            torch.cat([self.means.detach(), xyzs], dim=0)
        )
        self.gauss_params["scales"] = torch.nn.Parameter(
            torch.cat([self.scales.detach(), scales], dim=0)
        )
        self.gauss_params["quats"] = torch.nn.Parameter(
            torch.cat([self.quats.detach(), quats], dim=0)
        )
        self.gauss_params["opacities"] = torch.nn.Parameter(
            torch.cat([self.opacities.detach(), opacities], dim=0)
        )
        self.gauss_params["features_dc"] = torch.nn.Parameter(
            torch.cat([self.features_dc.detach(), features_dc], dim=0)
        )
        self.gauss_params["features_rest"] = torch.nn.Parameter(
            torch.cat([self.features_rest.detach(), features_rest], dim=0)
        )

        # Add the new parameters to the optimizer.
        for param_group, new_param in self.get_gaussian_param_groups().items():
            optimizer = optimizers.optimizers[param_group]
            old_param = optimizer.param_groups[0]["params"][0]
            param_state = optimizer.state[old_param]
            added_param_shape = (local_n_seed, *new_param[0].shape[1:])
            if "exp_avg" in param_state:
                param_state["exp_avg"] = torch.cat(
                    [
                        param_state["exp_avg"],
                        torch.zeros(added_param_shape).to(self.device),
                    ],
                    dim=0,
                )
            if "exp_avg_sq" in param_state:
                param_state["exp_avg_sq"] = torch.cat(
                    [
                        param_state["exp_avg_sq"],
                        torch.zeros(added_param_shape).to(self.device),
                    ],
                    dim=0,
                )

            del optimizer.state[old_param]
            optimizer.state[new_param[0]] = param_state
            optimizer.param_groups[0]["params"] = new_param
            del old_param

    def check_param_dims(self):
        Ng = self.means_copy.shape[0]
        return (Ng == self.rots_copy.shape[0]) and (Ng == self.scales_copy.shape[0])

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs_base = super().get_training_callbacks(training_callback_attributes)

        cb_seed = TrainingCallback(
            [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
            self.seed_cb,
            args=[
                training_callback_attributes.pipeline,
                training_callback_attributes.optimizers,
            ],
        )

        pose_buffer = TrainingCallback(
            [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
            training_callback_attributes.pipeline.datamanager.train_image_dataloader.update_dataset_from_buffer,
            args=[],
        )

        return [cb_seed, pose_buffer] + cbs_base

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        self.camera_optimizer.get_param_groups(param_groups=gps)

        if self.config.enable_semantics:
            # insert parameters for the CLIP Fields
            gps["clip_field"] = list(self.clip_field.parameters())

        return gps

    @torch.no_grad()
    def get_semantic_outputs(self, outputs: Dict[str, torch.Tensor]):
        if not self.training:
            # Normalize CLIP features rendered by feature field
            clip_features = outputs["clip"]
            clip_features /= clip_features.norm(dim=-1, keepdim=True)

            if self.viewer_utils.has_positives:
                if self.viewer_utils.has_negatives:
                    # Use paired softmax method as described in the paper with positive and negative texts
                    text_embs = torch.cat(
                        [self.viewer_utils.pos_embed, self.viewer_utils.neg_embed],
                        dim=0,
                    )

                    raw_sims = clip_features @ text_embs.T

                    # Broadcast positive label similarities to all negative labels
                    pos_sims, neg_sims = raw_sims[..., :1], raw_sims[..., 1:]
                    pos_sims = pos_sims.broadcast_to(neg_sims.shape)

                    # Updated Code
                    paired_sims = torch.cat(
                        (pos_sims.reshape((-1, 1)), neg_sims.reshape((-1, 1))), dim=-1
                    )

                    # compute the paired softmax
                    probs = paired_sims.softmax(dim=-1)[..., :1]
                    probs = probs.reshape((-1, neg_sims.shape[-1]))

                    torch.nan_to_num_(probs, nan=0.0)

                    sims, _ = probs.min(dim=-1, keepdim=True)
                    outputs["similarity"] = sims.reshape((*pos_sims.shape[:-1], 1))

                    # cosine similarity
                    outputs["raw_similarity"] = raw_sims[..., :1]
                else:
                    # positive embeddings
                    text_embs = self.viewer_utils.pos_embed

                    sims = clip_features @ text_embs.T
                    # Show the mean similarity if there are multiple positives
                    if sims.shape[-1] > 1:
                        sims = sims.mean(dim=-1, keepdim=True)
                    outputs["similarity"] = sims

                    # cosine similarity
                    outputs["raw_similarity"] = sims

                # for outputs similar to the GUI
                similarity_clip = outputs[f"similarity"] - outputs[f"similarity"].min()
                similarity_clip /= similarity_clip.max() + 1e-10
                outputs["similarity_GUI"] = apply_colormap(
                    similarity_clip, ColormapOptions("turbo")
                )

            if "rgb" in outputs.keys():
                if self.viewer_utils.has_positives:
                    # composited similarity
                    p_i = torch.clip(outputs["similarity"] - 0.5, 0, 1)

                    outputs["composited_similarity"] = apply_colormap(
                        p_i / (p_i.max() + 1e-6), ColormapOptions("turbo")
                    )
                    mask = (outputs["similarity"] < 0.5).squeeze()
                    outputs["composited_similarity"][mask, :] = outputs["rgb"][mask, :]

        return outputs

    # @torch.no_grad()
    def get_point_cloud_from_camera(
        self,
        camera: Cameras,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """Takes in a Camera and returns the back-projected points.

        Args:
            camera: Input Camera. This Camera Object should have all the
            needed information to compute the back-projected points.
            depth: Predicted depth image.

        Returns:
            back-projected points from the camera.
        """
        # camera intrinsics
        H, W, K = (
            camera.height.item(),
            camera.width.item(),
            camera.get_intrinsics_matrices(),
        )
        K = K.squeeze()

        # unnormalized pixel coordinates
        u_coords = torch.arange(W, device=self.device)
        v_coords = torch.arange(H, device=self.device)

        # meshgrid
        U_grid, V_grid = torch.meshgrid(u_coords, v_coords, indexing="xy")

        # transformed points in camera frame
        # [u, v, 1] = [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]] @ [x/z, y/z, 1]
        cam_pts_x = (U_grid - K[0, 2]) * depth.squeeze() / K[0, 0]
        cam_pts_y = (V_grid - K[1, 2]) * depth.squeeze() / K[1, 1]

        cam_pcd_points = torch.stack(
            (cam_pts_x, cam_pts_y, depth.squeeze(), torch.ones_like(cam_pts_y)), axis=-1
        ).to(self.device)

        # camera pose
        cam_pose = torch.eye(4, device=self.device)
        cam_pose[:3] = camera.camera_to_worlds

        # convert from OpenGL to OpenCV Convention
        cam_pose[:, 1] = -cam_pose[:, 1]
        cam_pose[:, 2] = -cam_pose[:, 2]

        # point = torch.einsum('ij,hkj->hki', cam_pose, cam_pcd_points)

        point = cam_pose @ cam_pcd_points.view(-1, 4).T
        point = point.T.view(*cam_pcd_points.shape[:2], 4)
        point = point[..., :3].view(*depth.shape[:2], 3)

        return point

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.
            compute_semantics: Option to compute the semantic information of the scene.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()),
                    int(camera.height.item()),
                    self.background_color,
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
            
            if self.param_delta is not None:
                param_delta_crop = self.param_delta[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

            if self.param_delta is not None:
                param_delta_crop = self.param_delta

        colors_crop = torch.cat(
            (features_dc_crop[:, None, :], features_rest_crop), dim=1
        )

        BLOCK_WIDTH = (
            16  # this controls the tile size of rasterization, 16 is a good default
        )
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(
                self.step // self.config.sh_degree_interval, self.config.sh_degree
            )
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze()
            sh_degree_to_use = None

        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        self.xys = info["means2d"]  # [1, N, 2]
        self.radii = info["radii"][0]  # [N]
        alpha = alpha[:, ...]

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(
                alpha > 0, depth_im, depth_im.detach().max()
            ).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        if self.param_delta is not None: 
            try:
                render_deltas, _, _ = rasterization(
                    means=means_crop,
                    quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
                    scales=torch.exp(scales_crop),
                    opacities=torch.sigmoid(opacities_crop).squeeze(-1),
                    colors=param_delta_crop.unsqueeze(1).expand(-1, 3),
                    viewmats=viewmat,  # [1, 4, 4]
                    Ks=K,  # [1, 3, 3]
                    width=W,
                    height=H,
                    packed=False,
                    near_plane=0.01,
                    far_plane=1e10,
                    render_mode=render_mode,
                    sh_degree=None,
                    sparse_grad=False,
                    absgrad=True,
                    rasterize_mode=self.config.rasterize_mode,
                    # set some threshold to disregrad small gaussians for faster rendering.
                    # radius_clip=3.0,
                )

                render_deltas = render_deltas[..., 0:1]

                # Normalize image
                render_deltas = ( render_deltas - render_deltas.min() ) / ( render_deltas.max() - render_deltas.min() )

            except:
                render_deltas = torch.zeros_like(rgb)
        else:
            render_deltas = torch.zeros_like(rgb)

        if self.config.enable_semantics:
            # generate a point cloud from the depth image
            pcd_points = self.get_point_cloud_from_camera(
                camera, depth_im.detach().clone()
            )
            # TODO: Just a temporary solution. Fix the static scaling factor.
            pcd_points *= self.config.semantic_field_scale

            camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

            # selected indices and points
            sel_idx = None

            # predicted CLIP embeddings
            clip_im = None

            if self.training:
                # subsample the points
                # number of points to subsample
                n_sub_sample = self.config.semantics_batch_size * 4096
                # n_sub_sample = pcd_points.view(-1, 3).shape[0]

                # get random samples
                sel_idx = torch.randperm(
                    pcd_points.view(-1, 3).shape[0], device=self.device
                )[:n_sub_sample]

                # selected points
                sel_pcd_points = pcd_points.view(-1, 3)[sel_idx]

                # predicted CLIP embeddings
                clip_im = self.clip_field(sel_pcd_points).float()
            elif self.config.enable_semantics:
                # predicted CLIP embeddings
                clip_im = (
                    self.clip_field(pcd_points.view(-1, 3))
                    .view(*depth_im.shape[:2], self.clip_embeds_input_dim)
                    .float()
                )

        outputs = {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
            "deltas": render_deltas.squeeze(0),
        }  # type: ignore

        if (
            self.config.output_semantics_during_training or not self.training
        ) and self.config.enable_semantics:
            # Compute semantic inputs, e.g., composited similarity.
            outputs["sel_idx"] = sel_idx
            outputs["clip"] = clip_im
            outputs = self.get_semantic_outputs(outputs=outputs)

        return outputs

    def depth_loss(self, outputs, depth_gt):
        """
        Compute the depth loss from the outputs and the ground truth depth image.

        Args:
            outputs: The output depth image from the model.
            depth_gt: The ground truth depth image.

        Returns:
            The depth loss.
        """
        # with torch.no_grad():
        #     depth_opp_mask = outputs["accumulation"] > self.config.depth_loss_oppac_thresh
        depth_val_mask = depth_gt > 0.1
        depth_mask = depth_val_mask  # & depth_opp_mask
        depth_pred = outputs["depth"]
        depth_loss = torch.nn.functional.l1_loss(
            depth_pred[depth_mask], depth_gt[depth_mask]
        )
        return depth_loss

    def get_loss_dict(
        self, outputs, batch, metrics_dict=None
    ) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        pred_img = outputs["rgb"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        if self.config.enable_semantics:
            if outputs["sel_idx"] is not None:
                # predicted CLIP embeddings
                pred_clip = outputs["clip"]

                # convert linear indices to row-column indices
                sel_idx_row, sel_idx_col = (
                    outputs["sel_idx"] // outputs["rgb"].shape[1],
                    outputs["sel_idx"] % outputs["rgb"].shape[1],
                )

                # scale factors
                scale_h = batch["clip"].shape[0] / outputs["rgb"].shape[0]
                scale_w = batch["clip"].shape[1] / outputs["rgb"].shape[1]

                # scaled indices
                sc_y_ind = (sel_idx_row * scale_h).long()
                sc_x_ind = (sel_idx_col * scale_w).long()

                # ground-truth CLIP embeddings
                gt_clip = batch["clip"][sc_y_ind, sc_x_ind, :].float()

                # mask using the alpha channel
                if self.config.enable_clip_mask:
                    if batch["image"].shape[-1] > 3:
                        # mask
                        mask = (
                            self.get_gt_img(batch["image"])[..., -1]
                            .float()[..., None]
                            .reshape(-1, 1)[outputs["sel_idx"]]
                        )
                        pred_clip = pred_clip * mask
                        gt_clip = gt_clip * mask

                # Loss: CLIP Embeddings
                clip_img_loss = self.config.clip_img_loss_weight * (
                    torch.nn.functional.mse_loss(pred_clip, gt_clip)
                    + (
                        1
                        - torch.nn.functional.cosine_similarity(
                            pred_clip, gt_clip, dim=-1
                        )
                    ).mean()
                )
            else:
                # Loss: CLIP Embeddings
                clip_img_loss = 0.0

        # RGB-related loss
        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(
            gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...]
        )
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        # main loss
        main_loss = (
            1 - self.config.ssim_lambda
        ) * Ll1 + self.config.ssim_lambda * simloss

        loss_dict = {
            "main_loss": main_loss,
            "scale_reg": scale_reg,
        }

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)

        if self.training and self.config.enable_depth_loss and "depth_image" in batch:
            depth_gt = self.get_gt_img(batch["depth_image"].unsqueeze(-1))
            dloss = self.depth_loss(outputs, depth_gt)
            loss_dict["depth_loss"] = dloss * self.config.depth_loss_mult

        if self.config.enable_semantics:
            loss_dict["clip_img_loss"] = clip_img_loss

        return loss_dict
