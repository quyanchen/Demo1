import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import functional as F
from tqdm import tqdm

def train_epoch(model, loader, optimizer, device, global_step=0, log_step=None):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        labels = batch["labels"].to(device)
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

        optimizer.zero_grad()
        logits = model(**inputs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        if log_step:
            log_step({"train/loss": loss.item(), "train/lr": optimizer.param_groups[0]["lr"]}, global_step)
        global_step += 1

    return total_loss / len(loader.dataset), global_step

@torch.no_grad()
def evaluate_epoch(model, loader, device, description="Evaluating"):
    model.eval()
    total_loss = 0.0
    all_labels, all_preds = [], []

    for batch in tqdm(loader, desc=description, leave=False):
        labels = batch["labels"].to(device)
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

        logits = model(**inputs)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)

        total_loss += loss.item() * labels.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
    }
