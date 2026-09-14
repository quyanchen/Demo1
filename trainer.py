import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from tqdm import tqdm

from metrics import ClassificationMetric
from utils import log_metrics, write_json


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader, device: torch.device,
                   description: str = "Evaluating") -> dict[str, Any]:
    model.eval()
    metric = ClassificationMetric(model.config.num_labels)
    total_loss, sample_count = 0.0, 0
    for batch in tqdm(loader, desc=description, leave=False):
        labels = batch.pop("labels").to(device)
        inputs = {k: v.to(device) for k, v in batch.items()}
        logits = model(**inputs)
        loss = F.cross_entropy(logits, labels)

        batch_size = labels.size(0)
        sample_count += batch_size
        total_loss += loss.item() * batch_size
        metric.update(logits.argmax(dim=-1), labels)

    if sample_count == 0:
        raise ValueError("Evaluation loader produced no samples")
    return {"loss": total_loss / sample_count, **metric.compute()}


class ClassificationTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Any,
        output_dir: Path,
        model_dir: Path,
        tokenizer: Any = None,
        run: Any = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        self.tokenizer = tokenizer
        self.run = run

        self.global_step = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_dev = None
        self.stale_epochs = 0
        self.history: list[dict[str, Any]] = []
        self.stop_reason = "max_epochs"

    def train_epoch(self, loader) -> tuple[float, int]:
        self.model.train()
        total_loss, sample_count = 0.0, 0
        for batch in tqdm(loader, desc="Training", leave=False):
            labels = batch.pop("labels").to(self.device)
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(self.model(**inputs), labels)
            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            sample_count += batch_size
            total_loss += loss.item() * batch_size
            self.global_step += 1

            if self.run is not None:
                self.run.log(
                    {"train/loss": loss.item(), "train/lr": self.optimizer.param_groups[0]["lr"]},
                    step=self.global_step,
                )

        if sample_count == 0:
            raise ValueError("Training loader produced no samples")
        return total_loss / sample_count, self.global_step

    def evaluate(self, loader, description: str = "Evaluating") -> dict[str, Any]:
        return evaluate_model(self.model, loader, self.device, description=description)

    def save_checkpoint(self) -> None:
        self.model.save_pretrained(self.model_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(self.model_dir)

    def fit(self, train_loader, dev_loader) -> dict[str, Any]:
        for epoch in range(1, self.config.epochs + 1):
            train_loss, _ = self.train_epoch(train_loader)
            dev_metrics = self.evaluate(dev_loader, description="Dev Eval")
            score = dev_metrics["macro_f1"]
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid development macro_f1: {score}")

            if self.best_score is None or score > self.best_score:
                self.best_score, self.best_epoch, self.best_dev = score, epoch, dev_metrics
                self.stale_epochs = 0
                self.save_checkpoint()
            else:
                self.stale_epochs += 1

            self.history.append({
                "epoch": epoch,
                "global_step": self.global_step,
                "train_loss": train_loss,
                "dev": dev_metrics,
            })
            write_json(self.output_dir / "history.json", {"epochs": self.history})
            log_metrics(self.run, "train", {"epoch_loss": train_loss, "epoch": epoch}, self.global_step)
            log_metrics(self.run, "dev", dev_metrics, self.global_step)
            print(f"Epoch {epoch}: loss={train_loss:.4f}, dev accuracy={dev_metrics['accuracy']:.4f}, macro-F1={score:.4f}")

            if self.config.patience and self.stale_epochs >= self.config.patience:
                self.stop_reason = "early_stopping"
                break

        if self.best_dev is None:
            raise ValueError("No development checkpoint was selected")

        return {
            "best_epoch": self.best_epoch,
            "epochs_completed": len(self.history),
            "stop_reason": self.stop_reason,
            "global_step": self.global_step,
            "dev": self.best_dev,
        }
