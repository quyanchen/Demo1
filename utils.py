import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import random
from dataclasses import asdict
from pathlib import Path

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def config_to_dict(config) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_info() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "swanlab")},
    }


def init_swanlab(config: dict, output_dir: Path):
    if config["swanlab_mode"] == "disabled":
        return None
    if config["swanlab_mode"] == "local" and importlib.util.find_spec("swanboard") is None:
        raise ImportError("SwanLab local dashboard requires swanboard; use --swanlab-mode offline or disabled")
    import swanlab

    return swanlab.init(
        project=config["swanlab_project"],
        name=config["run_id"],
        config=config,
        log_dir=str(output_dir / "swanlog"),
        mode=config["swanlab_mode"],
    )


def log_metrics(run, split: str, metrics: dict, step: int) -> None:
    if run is not None:
        scalars = {f"{split}/{key}": value for key, value in metrics.items() if isinstance(value, (int, float))}
        run.log(scalars, step=step)


def finish_swanlab(run) -> None:
    if run is not None:
        import swanlab

        swanlab.finish()
