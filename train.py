import argparse
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizerFast

from config import CONFIG, ExperimentConfig
from data import ToutiaoDataset
from model import BertClassifier
from trainer import ClassificationTrainer, evaluate_model
from utils import (
    config_to_dict, environment_info, file_sha256, finish_swanlab,
    init_swanlab, log_metrics, set_seed, write_json,
)


def create_run(config: ExperimentConfig) -> tuple[Path, Path]:
    config.validate()
    run_id = f"{config.run_name}-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    output_dir = config.output_root / run_id
    model_dir = config.model_output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir, model_dir


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
        trainer = ClassificationTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            config=config,
            output_dir=output_dir,
            model_dir=model_dir,
            tokenizer=tokenizer,
            run=run,
        )
        results = trainer.fit(train_loader, dev_loader)
        # 释放训练期占用的优化器状态与模型权重，防显存碎片
        del trainer, optimizer, model
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
        results["test"] = evaluate_model(best_model, test_loader, device, description="Test Eval")
        results["label_to_index"] = best_model.config.label2id
        log_metrics(run, "test", results["test"], results["global_step"])
        write_json(output_dir / "metrics.json", results)
        write_json(output_dir / "status.json", {"status": "complete"})
        print(f"Run: {output_dir}\nModel: {model_dir}\nTest accuracy: {results['test']['accuracy']:.4f}")
        del best_model, best_tokenizer
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
