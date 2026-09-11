import torch
from torch.nn import functional as F
from tqdm import tqdm

from metrics import ClassificationMetric


def train_epoch(model, loader, optimizer, device, global_step=0, log_step=None):
    model.train()
    total_loss, sample_count = 0.0, 0
    for batch in tqdm(loader, desc="Training", leave=False):
        labels = batch.pop("labels").to(device)
        inputs = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(**inputs), labels)
        loss.backward()
        optimizer.step()

        # 按实际 batch 样本数累计 loss，避免非整除末尾批次被错误赋权
        batch_size = labels.size(0)
        sample_count += batch_size
        total_loss += loss.item() * batch_size
        global_step += 1

        if log_step is not None:
            log_step({"train/loss": loss.item(), "train/lr": optimizer.param_groups[0]["lr"]}, global_step)

    if sample_count == 0:
        raise ValueError("Training loader produced no samples")
    return total_loss / sample_count, global_step


@torch.no_grad()
def evaluate_epoch(model, loader, device, description="Evaluating"):
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

