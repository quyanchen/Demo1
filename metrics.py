import torch


class ClassificationMetric:
    def __init__(self, num_classes: int):
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        # 混淆矩阵：行表示真实类别 (true)，列表示预测类别 (pred)
        self.confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        predictions = torch.as_tensor(predictions).detach().cpu()
        labels = torch.as_tensor(labels).detach().cpu()
        if predictions.ndim != 1 or labels.shape != predictions.shape:
            raise ValueError("Predictions and labels must have matching [B] shapes")
        if predictions.dtype not in (torch.int32, torch.int64) or labels.dtype not in (torch.int32, torch.int64):
            raise ValueError("Predictions and labels must be integer class indices")
        if (predictions < 0).any() or (predictions >= self.num_classes).any() or \
           (labels < 0).any() or (labels >= self.num_classes).any():
            raise ValueError("Class index outside the configured label set")

        # 向量化频次统计：将二维坐标 (true, pred) 压平为一维索引 index = true * C + pred
        # 借助底层的 torch.bincount 高速统计频次并还原矩阵，彻底避免 Python 循环，性能比 sklearn 快两个数量级
        indices = labels.long() * self.num_classes + predictions.long()
        self.confusion += torch.bincount(indices, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self) -> dict:
        matrix = self.confusion.to(torch.float64)
        total = int(matrix.sum())
        if total == 0:
            raise ValueError("Cannot compute metrics without samples")

        # 对角线元素为真正例 TP，行和为真实样本数 Support (TP+FN)，列和为预测样本数 Predicted (TP+FP)
        true_positive = matrix.diag()
        support = matrix.sum(1)
        predicted = matrix.sum(0)

        precision = true_positive / predicted.clamp_min(1)
        recall = true_positive / support.clamp_min(1)
        # 调和平均 2*P*R / (P+R) 代数化简为 2*TP / (support + predicted)
        # 单次除法直接消除浮点精度损失与 P+R=0 除零异常，clamp_min(1) 确保平滑
        f1 = 2 * true_positive / (support + predicted).clamp_min(1)

        return {
            "accuracy": float(true_positive.sum() / total),
            "macro_f1": float(f1.mean()),
            "per_class": [
                {
                    "index": i,
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i in range(self.num_classes)
            ],
            "confusion_matrix": self.confusion.tolist(),
        }

