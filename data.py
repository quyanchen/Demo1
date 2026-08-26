import json
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding

def load_label_mapping(data_dir):
    label_path = Path(data_dir) / "label2id.json"
    if not label_path.exists():
        raise FileNotFoundError(f"Label mapping file not found: {label_path.resolve()}. Please run prepare_data.py first.")
    with open(label_path, "r", encoding="utf-8") as f:
        label_to_index = json.load(f)
    index_to_label = {v: k for k, v in label_to_index.items()}
    return label_to_index, index_to_label

class ToutiaoDataset(Dataset):
    def __init__(self, path, tokenizer, max_length, label_to_index, use_keywords=False):
        with open(path, encoding="utf-8") as file:
            self.records = [json.loads(line) for line in file]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_index = label_to_index
        self.use_keywords = use_keywords

    def __len__(self):
        return len(self.records)

    def _get_text(self, record):
        text = record["text"]
        if self.use_keywords and record.get("keywords"):
            text = f"{text}。关键词：{record['keywords']}"
        return text

    def __getitem__(self, index):
        record = self.records[index]
        text = self._get_text(record)
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_length)
        encoded["label"] = self.label_to_index[record["label_id"]]
        return encoded

def build_loader(path, tokenizer, max_length, batch_size, shuffle, num_workers=0, *, label_to_index, use_keywords=False):
    dataset = ToutiaoDataset(path, tokenizer, max_length, label_to_index=label_to_index, use_keywords=use_keywords)
    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collator)
