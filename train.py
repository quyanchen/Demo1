import argparse
import math
import random
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from config import CONFIG, ROOT, ExperimentConfig

import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizerFast

from data import ToutiaoDataset
from engine import evaluate_epoch, train_epoch
from model import BertClassifier
from tracking import (
    config_to_dict, environment_info, file_sha256, finish_swanlab,
    init_swanlab, log_metrics, write_json,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_run(config: ExperimentConfig) -> tuple[Path, Path]:
    config.validate()
    run_id = f"{config.run_name}-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    output_dir = config.output_root / run_id
    model_dir = config.model_output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir, model_dir


def fit(model, tokenizer, train_loader, dev_loader, optimizer, device, config,
        output_dir: Path, model_dir: Path, run=None) -> dict:
    best_score, best_epoch, best_dev = None, 0, None
    global_step, stale_epochs = 0, 0
    history = []
    stop_reason = "max_epochs"

    def log_step(values, step):
        if run is not None:
            run.log(values, step=step)

    for epoch in range(1, config.epochs + 1):
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, device, global_step, log_step,
        )
        dev_metrics = evaluate_epoch(model, dev_loader, device, description="Dev Eval")
        score = dev_metrics["macro_f1"]
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid development macro_f1: {score}")
        if best_score is None or score > best_score:
            best_score, best_epoch, best_dev = score, epoch, dev_metrics
            stale_epochs = 0
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
        else:
            stale_epochs += 1

        history.append({"epoch": epoch, "global_step": global_step, "train_loss": train_loss, "dev": dev_metrics})
        write_json(output_dir / "history.json", {"epochs": history})
        log_metrics(run, "train", {"epoch_loss": train_loss, "epoch": epoch}, global_step)
        log_metrics(run, "dev", dev_metrics, global_step)
        print(f"Epoch {epoch}: loss={train_loss:.4f}, dev accuracy={dev_metrics['accuracy']:.4f}, macro-F1={score:.4f}")
        if config.patience and stale_epochs >= config.patience:
            stop_reason = "early_stopping"
            break

    if best_dev is None:
        raise ValueError("No development checkpoint was selected")
    return {
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "stop_reason": stop_reason, "global_step": global_step, "dev": best_dev,
    }


def run_experiment(config: ExperimentConfig) -> Path:
    output_dir, model_dir = create_run(config)
    run = None
    try:
        model_dir.mkdir(parents=True, exist_ok=False)
        set_seed(config.seed)
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if config.device == "auto" else torch.device(config.device)

        effective = {
            **config_to_dict(config),
            "run_id": output_dir.name,
            "output_dir": str(output_dir),
            "model_dir": str(model_dir),
            "resolved_device": str(device),
            "environment": environment_info(),
            "data_sha256": {},
        }
        write_json(output_dir / "config.json", effective)
        write_json(output_dir / "status.json", {"status": "running"})

        effective["data_sha256"] = {
            name: file_sha256(config.data_dir / name)
            for name in ("train.jsonl", "dev.jsonl", "label2id.json")
        }


        label_to_index = ToutiaoDataset.load_label_mapping(config.data_dir)
        bert_config = BertConfig.from_pretrained(
            config.model_name, revision=config.revision,
            cache_dir=config.model_cache, local_files_only=True,
        )
        if config.max_length > bert_config.max_position_embeddings:
            raise ValueError("max_length exceeds the model's position embeddings")
        bert_config.label2id = label_to_index
        bert_config.id2label = {index: label for label, index in label_to_index.items()}
        bert_config.classifier_dropout = config.dropout
        bert_config.text_max_length = config.max_length
        bert_config.use_keywords = config.use_keywords
        tokenizer = BertTokenizerFast.from_pretrained(
            config.model_name, revision=config.revision,
            cache_dir=config.model_cache, local_files_only=True,
        )
        train_dataset = ToutiaoDataset(
            config.data_dir / "train.jsonl", tokenizer, config.max_length,
            label_to_index, config.use_keywords,
        )
        dev_dataset = ToutiaoDataset(
            config.data_dir / "dev.jsonl", tokenizer, config.max_length,
            label_to_index, config.use_keywords,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, collate_fn=train_dataset.collate_fn,
        )
        dev_loader = DataLoader(
            dev_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, collate_fn=dev_dataset.collate_fn,
        )
        model = BertClassifier.from_pretrained(
            config.model_name, config=bert_config, revision=config.revision,
            cache_dir=config.model_cache, local_files_only=True,
        ).to(device)
        effective["resolved_revision"] = bert_config._commit_hash
        effective["num_labels"] = bert_config.num_labels
        effective["label_to_index"] = label_to_index
        effective["hardware"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        write_json(output_dir / "config.json", effective)

        run = init_swanlab(effective, output_dir)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
        )
        results = fit(
            model, tokenizer, train_loader, dev_loader, optimizer, device,
            config, output_dir, model_dir, run,
        )
        # 释放训练期占用的优化器状态与模型权重，防显存碎片
        del optimizer, model
        best_model = BertClassifier.from_pretrained(model_dir, local_files_only=True).to(device)
        best_tokenizer = BertTokenizerFast.from_pretrained(model_dir, local_files_only=True)
        test_path = config.data_dir / "test.jsonl"
        effective["data_sha256"][test_path.name] = file_sha256(test_path)
        write_json(output_dir / "config.json", effective)

        test_dataset = ToutiaoDataset(
            test_path, best_tokenizer, best_model.config.text_max_length,
            best_model.config.label2id, best_model.config.use_keywords,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, collate_fn=test_dataset.collate_fn,
        )
        results["test"] = evaluate_epoch(best_model, test_loader, device, description="Test Eval")
        results["label_to_index"] = best_model.config.label2id
        log_metrics(run, "test", results["test"], results["global_step"])
        write_json(output_dir / "metrics.json", results)
        write_json(output_dir / "status.json", {"status": "complete"})
        print(f"Run: {output_dir}\nModel: {model_dir}\nTest accuracy: {results['test']['accuracy']:.4f}")
        return output_dir
    except BaseException as error:
        write_json(output_dir / "status.json", {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        finish_swanlab(run)


def parse_args(argv=None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Train a handwritten BERT classifier using local model files.")
    parser.add_argument("--data-dir", type=Path, default=CONFIG.data_dir)
    parser.add_argument("--output-root", type=Path, default=CONFIG.output_root)
    parser.add_argument("--model-output-root", type=Path, default=CONFIG.model_output_root)
    parser.add_argument("--model-name", default=CONFIG.model_name)
    parser.add_argument("--revision", default=CONFIG.revision)
    parser.add_argument("--run-name", default=CONFIG.run_name)
    parser.add_argument("--epochs", type=int, default=CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    parser.add_argument("--learning-rate", type=float, default=CONFIG.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=CONFIG.weight_decay)
    parser.add_argument("--dropout", type=float, default=CONFIG.dropout)
    parser.add_argument("--max-length", type=int, default=CONFIG.max_length)
    parser.add_argument("--num-workers", type=int, default=CONFIG.num_workers)
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    parser.add_argument("--patience", type=int, default=CONFIG.patience, help="Non-improving epochs; 0 disables early stopping")
    parser.add_argument("--use-keywords", action="store_true", default=CONFIG.use_keywords)
    parser.add_argument("--swanlab-mode", choices=("offline", "local", "online", "disabled"), default=CONFIG.swanlab_mode)
    parser.add_argument("--device", default=CONFIG.device, help="auto, cpu, cuda, cuda:0, ...")
    return ExperimentConfig(**vars(parser.parse_args(argv)))


def main(argv=None):
    return run_experiment(parse_args(argv))


if __name__ == "__main__":
    main()
