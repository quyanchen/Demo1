import argparse
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from config import MODEL_ROOT

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from data import ToutiaoDataset
from engine import evaluate_epoch
from model import BertClassifier
from utils import file_sha256, write_json


def evaluate_run(run_dir: Path, data_file: Path | None = None,
                 batch_size: int | None = None, device: str = "auto") -> dict:
    run_dir = Path(run_dir).resolve()
    saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    model_dir = Path(saved["model_dir"])
    if not model_dir.resolve().is_relative_to(MODEL_ROOT.resolve()):
        raise ValueError(f"Saved model directory must be inside {MODEL_ROOT}")
    batch_size = saved["batch_size"] if batch_size is None else batch_size
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    selected_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if device == "auto" else torch.device(device)
    model = BertClassifier.from_pretrained(model_dir, local_files_only=True).to(selected_device)
    tokenizer = BertTokenizerFast.from_pretrained(model_dir, local_files_only=True)
    path = Path(data_file) if data_file is not None else Path(saved["data_dir"]) / "test.jsonl"
    dataset = ToutiaoDataset(
        path, tokenizer, model.config.text_max_length,
        model.config.label2id, model.config.use_keywords,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn,
    )
    metrics = evaluate_epoch(model, loader, selected_device, description="Test Eval")
    fingerprint = file_sha256(path)
    expected = saved["data_sha256"].get("test.jsonl") if data_file is None else None
    report = {
        "data_file": str(path.resolve()),
        "data_sha256": fingerprint,
        "matches_training_test_data": fingerprint == expected if expected is not None else None,
        "batch_size": batch_size,
        "device": str(selected_device),
        "label_to_index": model.config.label2id,
        "metrics": metrics,
    }
    evaluation_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    output = run_dir / f"evaluation-{evaluation_id}.json"
    write_json(output, report)
    print(f"Accuracy: {metrics['accuracy']:.4f}, macro-F1: {metrics['macro_f1']:.4f}\nResult: {output}")
    return report



def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the complete local model selected by a training run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    return evaluate_run(args.run_dir, args.data_file, args.batch_size, args.device)


if __name__ == "__main__":
    main()
