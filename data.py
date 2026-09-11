import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ToutiaoDataset(Dataset):

    def __init__(self, path, tokenizer, max_length, label_to_index, use_keywords=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_index = dict(label_to_index)
        self.use_keywords = use_keywords
        self.records = self.load(Path(path))

    @staticmethod
    def load_label_mapping(data_dir: Path) -> dict[str, int]:
        path = Path(data_dir) / "label2id.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run prepare_data.py first.")
        return json.loads(path.read_text(encoding="utf-8"))

    def load(self, path: Path) -> list[dict]:
        records = []
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                    if not record.get("text", "").strip():
                        raise ValueError("text must not be blank")
                    if record["label_id"] not in self.label_to_index:
                        raise ValueError(f"Unknown label: {record['label_id']}")
                except Exception as error:
                    raise ValueError(f"{path}:{number}: {error}")
                records.append(record)
        if not records:
            raise ValueError(f"{path}: dataset is empty")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _get_text(self, record: dict) -> str:
        text = record["text"]
        if self.use_keywords and record.get("keywords"):
            text = f"{text}。关键词：{record['keywords']}"
        return text

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        return {"text": self._get_text(record), "label": self.label_to_index[record["label_id"]]}

    def collate_fn(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        if not samples:
            raise ValueError("Cannot collate an empty batch")
        batch = self.tokenizer(
            [sample["text"] for sample in samples],
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor([sample["label"] for sample in samples], dtype=torch.long)
        return dict(batch)
