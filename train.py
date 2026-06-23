import argparse
import random
from pathlib import Path

import swanlab
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import TextDataset, build_label_mapping, load_jsonl
from model import TextClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="BERT text classification")
    parser.add_argument("--model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--train_path", type=str, default="dataset/train_3k.jsonl")
    parser.add_argument("--dev_path", type=str, default="dataset/dev_1k.jsonl")
    parser.add_argument("--test_path", type=str, default="dataset/test_1k.jsonl")
    parser.add_argument("--save_path", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_keywords", action="store_true")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(model, data_loader, optimizer, args):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for batch in data_loader:
        batch = {key: value.to(args.device) for key, value in batch.items()}
        labels = batch.pop("labels")

        optimizer.zero_grad()
        logits = model(**batch)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, data_loader, args):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for batch in data_loader:
        batch = {key: value.to(args.device) for key, value in batch.items()}
        labels = batch.pop("labels")

        logits = model(**batch)
        loss = F.cross_entropy(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def main(args):
    set_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"Using device: {args.device}")

    train_samples = load_jsonl(args.train_path)
    dev_samples = load_jsonl(args.dev_path)
    test_samples = load_jsonl(args.test_path)

    args.label_to_id, args.id_to_label = build_label_mapping(train_samples)
    args.num_classes = len(args.label_to_id)

    swanlab.init(
        project="demo1-text-classification",
        experiment_name=f"bert-lr{args.learning_rate}-bs{args.batch_size}",
        config=vars(args),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = TextDataset(train_samples, tokenizer, args)
    dev_dataset = TextDataset(dev_samples, tokenizer, args)
    test_dataset = TextDataset(test_samples, tokenizer, args)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    dev_loader = DataLoader(dev_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = TextClassifier(args).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    best_dev_accuracy = -1.0

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train(model, train_loader, optimizer, args)
        dev_loss, dev_accuracy = evaluate(model, dev_loader, args)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
            f"Dev Loss: {dev_loss:.4f} | Dev Acc: {dev_accuracy:.4f}"
        )

        swanlab.log(
            {
                "train/loss": train_loss,
                "train/accuracy": train_accuracy,
                "dev/loss": dev_loss,
                "dev/accuracy": dev_accuracy,
            },
            step=epoch,
        )

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args).copy(),
                    "epoch": epoch,
                    "best_dev_accuracy": best_dev_accuracy,
                },
                args.save_path,
            )
            print(f"Best checkpoint saved to {args.save_path}")

    checkpoint = torch.load(args.save_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_accuracy = evaluate(model, test_loader, args)
    swanlab.log({"test/loss": test_loss, "test/accuracy": test_accuracy})

    print(
        f"Best Epoch: {checkpoint['epoch']} | "
        f"Best Dev Acc: {checkpoint['best_dev_accuracy']:.4f} | "
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f}"
    )


if __name__ == "__main__":
    main(parse_args())