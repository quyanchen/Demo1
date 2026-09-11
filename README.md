# Demo 1: Chinese News Text Classification with BERT

This repository provides a clean, modular PyTorch implementation of Chinese text classification on a 15-class subset of the Toutiao news dataset using [`google-bert/bert-base-chinese`](https://huggingface.co/google-bert/bert-base-chinese). 



## Revision Notes & Changelog (v2.0)


本次核心架构重构与优化如下：
- **数据目录规范化**：将数据集统一收归至 `dataset/toutiao/` 目录（区分 `raw/` 与 `processed/`），清理根目录散落文件。
- **手写动态批次对齐**：在 `ToutiaoDataset` 中实现自定义 `collate_fn`，按批次最大长度动态 Padding。
- **手写任务分类头**：继承 `BertPreTrainedModel`，直接切片提取最后一层原生 `[CLS]`（`last_hidden_state[:, 0]`）直连线性分类头，规避预训练 NSP pooler 的 $\tanh$ 梯度饱和，保留 Hugging Face 权重保存与加载规范。
- **向量化指标**：借助 `torch.bincount` 无循环快速累积混淆矩阵。
- 按实际样本量加权累计 Epoch Loss，消除尾批次失真；显式执行 `del optimizer, model` 释放显存，并从磁盘重载 Checkpoint 执行测试集推断。

---

## Project Structure

```text
.
├── config.py           # Experiment configuration dataclass and validation
├── data.py             # ToutiaoDataset and custom dynamic-padding collate_fn
├── model.py            # BertClassifier inheriting from BertPreTrainedModel
├── metrics.py          # ClassificationMetric with bincount-based confusion matrix & Macro-F1
├── engine.py           # train_epoch and evaluate_epoch routines
├── train.py            # Training orchestration, early stopping, and test inference
├── evaluate.py         # Standalone offline evaluation entry point
├── prepare_data.py     # Raw text parsing, label mapping, and data leakage audit
├── tracking.py         # Atomic JSON persistence and SwanLab experiment logging
├── dataset/
│   └── toutiao/
│       ├── raw/        # Raw split files (train_3k.txt, dev_1k.txt, test_1k.txt)
│       └── processed/  # Processed JSONL files, mappings, and audit report
├── tests/              # Comprehensive regression test suite
├── requirements.txt    # Environment dependencies
└── README.md
```

---

## Requirements

* Python >= 3.10
* PyTorch >= 2.3.0
* Transformers >= 4.45.2

Install the required packages:
```bash
pip install -r requirements.txt
```

---

## Quickstart

### 1. Data Preparation
To rebuild standard JSONL files and the bijection label mapping from raw text:
```bash
python prepare_data.py
```
This script derives the continuous label mapping strictly from the training split (preventing label leakage) and runs a title overlap audit across splits, outputting `dataset/toutiao/processed/audit.json`.

### 2. Model Training
```bash
python train.py --run-name baseline
```
* **Default hyperparameters**: `batch_size=32`, `lr=2e-5`, `weight_decay=0.01`, `dropout=0.1`, `max_length=64`, `epochs=5`, `patience=3`.
* Evaluates on `test.jsonl` upon completion and exports all configs, metrics, and logs to `outputs/<run_id>/`.

### 3. Evaluation
To independently evaluate a previously saved experiment checkpoint on the test set:
```bash
python evaluate.py --run-dir outputs/<your_run_id>
```

## Results

Evaluation on the 15-class Toutiao news dataset (Train: 3,000 / Dev: 1,000 / Test: 1,064):

| Split | Accuracy | Macro-F1 | Samples | Classes |
| :--- | :---: | :---: | :---: | :---: |
| **Dev** | 84.10% | 82.84% | 1,000 | 15 |
| **Test** | 82.71% | 81.73% | 1,064 | 15 |
