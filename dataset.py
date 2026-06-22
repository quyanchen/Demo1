import json

import torch
from torch.utils.data import Dataset


def load_jsonl(file_path):
    samples = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            samples.append(json.loads(line))

    return samples


def build_label_mapping(samples):

    labels = sorted(
        {sample["label"] for sample in samples},
        key=int,
    )

    label_to_id = {
        label: index
        for index, label in enumerate(labels)
    }

    id_to_label = {
        index: label
        for label, index in label_to_id.items()
    }

    return label_to_id, id_to_label


class TextDataset(Dataset):

    def __init__(
        self,
        samples,
        tokenizer,
        label_to_id,
        max_length=128,
        use_keywords=False,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
        self.use_keywords = use_keywords

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        text = sample["text"]
        keywords = sample["keywords"]
        label = self.label_to_id[sample["label"]]

        if self.use_keywords and keywords:
            encoding = self.tokenizer(
                text,
                text_pair=keywords,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        else:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

        item["labels"] = torch.tensor(
            label,
            dtype=torch.long,
        )

        return item