from typing import Union

import torch
import yaml
from pathlib import Path
from typing import Tuple, Any
from copy import deepcopy

from nerfbridge.ros_pipeline import ROSPipeline

import cv2
from rich.console import Console
from rich.progress import Progress


def load_from_checkpoint(
    config_path: Path, checkpoint_path: Path
) -> Tuple[Any, ROSPipeline]:
    """Load a ROSPipeline from a checkpoint for evaluation and viewing."""

    # load save config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)

    # load checkpoints from wherever they were saved
    config.load_dir = config.get_checkpoint_dir()

    config.data = config_path.parent / "dataparser_config.json"
    config.pipeline.datamanager.data = config.data

    loaded_state = torch.load(checkpoint_path, map_location="cpu")

    # setup pipeline (which includes the DataManager)
    cache_dir = config_path.parent / "run_data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(
        cache_dir=cache_dir, device=device, run_mode="eval", test_mode="inference"
    )
    pipeline.eval()

    pipeline.load_pipeline(
        loaded_state["pipeline"], loaded_state["data_states"], loaded_state["step"]
    )

    return config, pipeline


def eval_on_cache(pipeline: ROSPipeline):
    """Evaluate the model on the eval data cached during training."""
    console = Console(width=80)
    base_camera = pipeline.datamanager.train_dataset.cameras[0:1]

    # Load the cached data
    rgbs_filenames = list(
        (pipeline.datamanager.cache_dir / "eval" / "rgb").glob("*.png")
    )
    rgbs_filenames.sort()

    console.rule("Loading Eval Images")
    console.print(f"Loading {len(rgbs_filenames)} images from cache")
    rgbs = []
    with Progress() as progress:
        task = progress.add_task("Loading images...", total=len(rgbs_filenames))
        for fname in rgbs_filenames:
            rgb = cv2.imread(str(fname))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) / 255.0
            rgbs.append(torch.from_numpy(rgb).float())
            progress.update(task, advance=1)

    eval_c2ws = pipeline.data_states["eval_c2ws"]
    console.print(f"Loaded {len(rgbs)} images and {len(eval_c2ws)} poses!")
    console.print(
        "If these numbers don't match then it's because the checkpointing is slow!"
    )
    console.print("We can fix this later! -- Javier")

    console.rule("Evaluating")
    eval_metrics = []
    with Progress() as progress:
        task = progress.add_task("Evaluating images...", total=len(eval_c2ws))
        for idx in range(len(eval_c2ws)):
            batch = {"image": rgbs[idx].cuda()}

            camera = deepcopy(base_camera)
            c2w = eval_c2ws[idx].unsqueeze(0).cuda()
            camera.camera_to_worlds = c2w

            with torch.no_grad():
                outputs = pipeline.model.get_outputs(camera=camera)
                metrics, _ = pipeline.model.get_image_metrics_and_images(outputs, batch)
                eval_metrics.append(metrics)
            progress.update(task, advance=1)

    for key in eval_metrics[0].keys():
        average_metric = torch.tensor([m[key] for m in eval_metrics]).mean().item()
        console.print(f"{key}: {average_metric:.4f}")

    return eval_metrics
