import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    raw_data_dir: Path = Path(".")
    rebuilt_data_dir: Path = Path("processed_rebuilt")
    data_dir: Path = Path("processed")
    model_name: str = "google-bert/bert-base-chinese"
    model_cache: Optional[Path] = None
    output_dir: Path = Path("outputs/baseline")
    test_file: Path = Path("processed/test_1k.jsonl")
    run_name: str = "baseline"
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    dropout: float = 0.1
    max_length: int = 64
    num_workers: int = 0
    seed: int = 42
    patience: int = 3
    use_keywords: bool = False
    swanlab_mode: str = "local"
    swanlab_project: str = "demo1-text-classification"


def parse_config():
    defaults = Config()
    cache = os.environ.get("DEMO1_MODEL_CACHE")

    parser = argparse.ArgumentParser(description="BERT text classification")
    parser.add_argument("--raw-data-dir", type=Path, default=defaults.raw_data_dir)
    parser.add_argument("--rebuilt-data-dir", type=Path, default=defaults.rebuilt_data_dir)
    parser.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    parser.add_argument("--model-name", default=defaults.model_name)
    parser.add_argument("--model-cache", type=Path, default=Path(cache) if cache else None)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--test-file", type=Path, default=defaults.test_file)
    parser.add_argument("--run-name", default=defaults.run_name)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--max-length", type=int, default=defaults.max_length)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--use-keywords", action="store_true", default=defaults.use_keywords)
    parser.add_argument("--swanlab-mode", default=defaults.swanlab_mode)
    parser.add_argument("--swanlab-project", default=defaults.swanlab_project)

    return Config(**vars(parser.parse_args()))
