import torch
from transformers import AutoTokenizer

from config import parse_config
from data import build_loader
from engine import evaluate_epoch
from model import BertClassifier


def main():
    args = parse_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.output_dir / "best.pt"
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], cache_dir=args.model_cache)
    loader = build_loader(
        args.test_file, tokenizer, config["max_length"], args.batch_size, False, args.num_workers,
        label_to_index=checkpoint["label_to_index"], use_keywords=config["use_keywords"],
    )
    model = BertClassifier(
        config["model_name"],
        len(checkpoint["label_to_index"]),
        config["dropout"],
        args.model_cache,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    print(f"Evaluating on {args.test_file}...")
    metrics = evaluate_epoch(model, loader, device, description="Test Eval")

    print("\nEvaluation Results:")
    print(f"  Loss:     {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro-F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
