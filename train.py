import json
import random

import swanlab
import torch
from transformers import AutoTokenizer

from config import parse_config
from data import build_loader, load_label_mapping
from engine import evaluate_epoch, train_epoch
from model import BertClassifier
from tracking import config_to_dict, init_swanlab, log_metrics


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_config()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.model_cache is not None:
        args.model_cache.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    label_to_index, _ = load_label_mapping(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.model_cache)

    print("Loading datasets...")
    train_loader = build_loader(
        args.data_dir / "train_3k.jsonl", tokenizer, args.max_length, args.batch_size, True, args.num_workers,
        label_to_index=label_to_index, use_keywords=args.use_keywords,
    )
    dev_loader = build_loader(
        args.data_dir / "dev_1k.jsonl", tokenizer, args.max_length, args.batch_size, False, args.num_workers,
        label_to_index=label_to_index, use_keywords=args.use_keywords,
    )

    model = BertClassifier(args.model_name, len(label_to_index), args.dropout, args.model_cache).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    run = init_swanlab(args)
    global_step = 0
    best_accuracy = 0.0
    best_epoch = 0
    best_dev_metrics = None
    checkpoint_path = args.output_dir / "best.pt"
    patience = args.patience
    patience_counter = 0

    def log_step(values, step):
        run.log(values, step=step)

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, global_step = train_epoch(model, train_loader, optimizer, device, global_step, log_step)
        dev_metrics = evaluate_epoch(model, dev_loader, device, description="Dev Eval")

        run.log({"train/epoch_loss": train_loss}, step=epoch)
        log_metrics(run, "dev", dev_metrics, epoch)

        print(
            f"Epoch [{epoch}/{args.epochs}] - "
            f"Train Loss: {train_loss:.4f} | "
            f"Dev Loss: {dev_metrics['loss']:.4f} | "
            f"Dev Acc: {dev_metrics['accuracy']:.4f} | "
            f"Dev Macro-F1: {dev_metrics['macro_f1']:.4f}"
        )

        if dev_metrics["accuracy"] > best_accuracy:
            best_accuracy = dev_metrics["accuracy"]
            best_epoch = epoch
            best_dev_metrics = dev_metrics
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config_to_dict(args),
                    "label_to_index": label_to_index,
                    "best_epoch": best_epoch,
                    "dev_metrics": best_dev_metrics,
                },
                checkpoint_path,
            )
            print(f"Saved new best model (Dev Acc: {best_accuracy:.4f}) to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience is not None and patience_counter >= patience:
                print(f"Early stopping triggered: validation accuracy did not improve for {patience} epochs.")
                break

    print("\nEvaluating best checkpoint on test set...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_loader = build_loader(
        args.data_dir / "test_1k.jsonl", tokenizer, args.max_length, args.batch_size, False, args.num_workers,
        label_to_index=label_to_index, use_keywords=args.use_keywords,
    )
    test_metrics = evaluate_epoch(model, test_loader, device, description="Test Eval")
    log_metrics(run, "test", test_metrics, best_epoch)

    print("Test Results:")
    print(f"  Loss:     {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Macro-F1: {test_metrics['macro_f1']:.4f}")

    results = {
        "best_epoch": best_epoch,
        "dev": best_dev_metrics,
        "test": test_metrics,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    swanlab.finish()


if __name__ == "__main__":
    main()
