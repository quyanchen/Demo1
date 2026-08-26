# Demo1

A simple PyTorch implementation of Chinese news classification using [`bert-base-chinese`](https://huggingface.co/google-bert/bert-base-chinese). The code includes data preprocessing, a custom classification head, training and evaluation loops, checkpoint selection, and SwanLab logging.

The dataset is a 15-class subset of the [Toutiao text classification dataset](https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset), with 3,000 training examples, 1,000 development examples, and 1,064 test examples.

## Results

| Split | Accuracy | Macro-F1 |
|---|---:|---:|
| Dev | 84.10% | 82.84% |
| Test | 81.86% | 79.50% |

The baseline uses seed 42, batch size 32, learning rate `2e-5`, maximum length 64, and five training epochs. The best checkpoint is selected by development accuracy.

## Usage

Install the dependencies:

```bash
pip install -r requirements.txt
```

The processed data is already included. To rebuild it from the raw files:

```bash
python prepare_data.py
```

Train and evaluate the model:

```bash
python train.py
python evaluate.py
```

Run the tests:

```bash
python -m unittest discover -v
```

Hyperparameters and paths are defined in `config.py`. Model files are stored in `D:\1Code\1Model\huggingface` by default; set `DEMO1_MODEL_CACHE` to use another directory. Training outputs are written to `outputs/baseline/`, including `best.pt`, `metrics.json`, and SwanLab logs.

## Files

```text
prepare_data.py   Data conversion
data.py           Dataset and data loaders
model.py          BERT classifier
engine.py         Training and evaluation loops
train.py          Training entry point
evaluate.py       Evaluation entry point
tracking.py       SwanLab logging
config.py         Experiment settings
tests/            Unit tests
```
