import os
from pathlib import Path
from types import SimpleNamespace

MODEL_CACHE = Path(os.environ.get("DEMO1_MODEL_CACHE", r"D:\1Code\1Model\huggingface"))
os.environ["HF_HOME"] = str(MODEL_CACHE)
os.environ["HF_HUB_CACHE"] = str(MODEL_CACHE / "hub")
os.environ["HF_XET_CACHE"] = str(MODEL_CACHE / "xet")

CONFIG = SimpleNamespace(
    raw_data_dir=Path("."),
    rebuilt_data_dir=Path("processed_rebuilt"),
    data_dir=Path("processed"),
    model_name="google-bert/bert-base-chinese",
    model_cache=MODEL_CACHE,
    output_dir=Path("outputs/baseline"),
    test_file=Path("processed/test_1k.jsonl"),
    run_name="baseline",
    epochs=5,
    batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    dropout=0.1,
    max_length=64,
    num_workers=0,
    seed=42,
    patience=3,
    use_keywords=False,
    swanlab_mode="local",
    swanlab_project="demo1-text-classification",
)
