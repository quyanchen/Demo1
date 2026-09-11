"""Experiment settings and explicit model-cache policy."""

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path(r"D:\1Code\1Model")
MODEL_CACHE = Path(os.environ.get("DEMO1_MODEL_CACHE", MODEL_ROOT / "huggingface")).resolve()
if not MODEL_CACHE.is_relative_to(MODEL_ROOT.resolve()):
    raise ValueError("DEMO1_MODEL_CACHE must be inside D:\\1Code\\1Model")

os.environ["HF_HOME"] = str(MODEL_CACHE)
os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE)
os.environ["HF_XET_CACHE"] = str(MODEL_CACHE / "xet")


@dataclass
class ExperimentConfig:
    data_dir: Path = ROOT / "dataset" / "toutiao" / "processed"
    output_root: Path = ROOT / "outputs"
    model_output_root: Path = MODEL_ROOT / "demo1"
    model_name: str = "google-bert/bert-base-chinese"
    revision: str = "8f23c25b06e129b6c986331a13d8d025a92cf0ea"
    model_cache: Path = MODEL_CACHE
    run_name: str = "baseline"
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    dropout: float = 0.1
    max_length: int = 64
    num_workers: int = 0
    seed: int = 42
    patience: int = 3  # Zero disables early stopping.
    use_keywords: bool = False
    swanlab_mode: str = "offline"
    swanlab_project: str = "demo1-text-classification"
    device: str = "auto"

    def __post_init__(self) -> None:
        for name in ("data_dir", "output_root", "model_output_root", "model_cache"):
            setattr(self, name, Path(getattr(self, name)).resolve())
        if Path(self.model_name).is_dir():
            self.model_name = str(Path(self.model_name).resolve())

    def validate(self) -> None:
        for name in ("epochs", "batch_size", "max_length"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_length < 2:
            raise ValueError("max_length must allow BERT's CLS and SEP tokens")
        if self.num_workers < 0 or self.patience < 0:
            raise ValueError("num_workers and patience must be nonnegative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.run_name) or ".." in self.run_name or self.run_name.startswith((".", "-")):
            raise ValueError("run_name must contain only letters, digits, '.', '-' or '_' and cannot start with '.' or '-'")
        if not self.output_root.resolve().is_relative_to(ROOT):
            raise ValueError(f"output_root must stay inside {ROOT}")
        if not self.model_cache.resolve().is_relative_to(MODEL_ROOT.resolve()):
            raise ValueError(f"model_cache must be inside {MODEL_ROOT}")
        if not self.model_output_root.resolve().is_relative_to(MODEL_ROOT.resolve()):
            raise ValueError(f"model_output_root must be inside {MODEL_ROOT}")


CONFIG = ExperimentConfig()
RAW_DATA_DIR = ROOT / "dataset" / "toutiao" / "raw"
