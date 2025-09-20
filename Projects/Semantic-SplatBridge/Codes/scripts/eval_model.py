from pathlib import Path
import os
from dataclasses import dataclass

from rich.console import Console
import json
import tyro

from nerfbridge.eval import load_from_checkpoint, eval_on_cache

CONSOLE = Console()

@dataclass
class RunEval:
    """Configuration for evaluation."""

    load_config: Path = Path("outputs/latest_run/config.yml")
    """Path to config file."""

    def main(self) -> None:
        """Main function."""

        # Path to checkpoint file
        checkpoint_dir = self.load_config.parent / "nerfstudio_models"
        load_step = sorted(
            int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(checkpoint_dir)
        )[-1]
        latest_checkpoint = checkpoint_dir / f"step-{load_step:09d}.ckpt"
        CONSOLE.print(f"Loading checkpoint from {latest_checkpoint}")

        # Load the pipeline
        config, pipeline = load_from_checkpoint(self.load_config, latest_checkpoint)
        pipeline.eval()

        eval_metrics = eval_on_cache(pipeline)

        # Save the evaluation metrics
        eval_metrics_path = self.load_config.parent / "eval_metrics.json"
        with open(str(eval_metrics_path), "w") as f:
            json.dump(eval_metrics, f, indent=4)

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[RunEval]).main()


if __name__ == "__main__":
    entrypoint()
