from typing import Union

from nerfbridge.features.clip import clip
from torchvision.transforms import Compose, Resize, Normalize
from einops import rearrange
import torch
from time import perf_counter
import pdb


class CLIPWrapper:
    def __init__(self, h_in: int, w_in: int, device="cuda"):
        self.model, preprocess = clip.load("ViT-L/14@336px", device=device)
        # Keep only the Resize and Normalize transforms
        self.preprocess = Compose(
            [t for t in preprocess.transforms if isinstance(t, (Resize, Normalize))]
        )
        self.preprocess.transforms[0].antialias = False

        with torch.no_grad():
            dim_input = torch.rand(1, 3, h_in, w_in)
            dim_pp = self.preprocess(dim_input)
            dim_output = self.model.get_patch_encodings(dim_pp.to(device))
            h_pp, w_pp = dim_pp.shape[-2:]

        self.out_w = w_pp // self.model.visual.patch_size
        self.out_h = h_pp // self.model.visual.patch_size
        self.feature_dim = dim_output.shape[-1]
        self.feature_type = "CLIP"

        self.device = device

    def get_features(self, image: torch.Tensor):
        """
        Get patch image features from CLIP for a single image.
        Args:
            image (torch.Tensor): Image tensor of shape (H, W, C)
            or (B, H, W, C).
        """
        # Check for batch dimension
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        image = rearrange(image, "b h w c -> b c h w")
        processed_image = self.preprocess(image).to(self.device)

        with torch.no_grad():
            features = self.model.get_patch_encodings(processed_image)

        features = rearrange(
            features, "b (h w) c -> b h w c", h=self.out_h, w=self.out_w
        )

        return features.squeeze()  # get rid of batch dimension if it exists


if __name__ == "__main__":
    model = CLIPWrapper(720, 1280, "cuda")
    image = torch.rand(720, 1280, 3)

    t_avg = 0
    for _ in range(10):
        t0 = perf_counter()
        features = model.get_features(image)
        t1 = perf_counter()
        t_avg += t1 - t0
    t_avg /= 10

    print(f"Time taken: {t_avg:.3f}s or {1/t_avg:.3f}Hz")
    pdb.set_trace()
