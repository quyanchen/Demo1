import json

from config import CONFIG

SPLITS = {
    "train_3k.txt": "train_3k.jsonl",
    "dev_1k.txt": "dev_1k.jsonl",
    "test_1k.txt": "test_1k.jsonl",
}

def parse_line(line):
    article_id, label_id, category, text, keywords = line.rstrip("\n").split("_!_", 4)
    return {"id": article_id, "label_id": label_id, "category": category, "text": text, "keywords": keywords}

def convert_file(source, destination):
    count, labels = 0, set()
    with source.open(encoding="utf-8") as input_file, destination.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        for line in input_file:
            record = parse_line(line)
            labels.add(record["label_id"])
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count, labels

def main():
    target_dirs = [CONFIG.rebuilt_data_dir, CONFIG.data_dir]
    for d in target_dirs:
        d.mkdir(parents=True, exist_ok=True)

    train_labels = None
    for source_name, destination_name in SPLITS.items():
        count, labels = convert_file(
            CONFIG.raw_data_dir / source_name,
            CONFIG.rebuilt_data_dir / destination_name,
        )
        if "train" in source_name:
            train_labels = sorted(list(labels))
        print(f"{destination_name}: {count} rows, {len(labels)} unique labels")

    label2id = {label: idx for idx, label in enumerate(train_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    for d in target_dirs:
        with (d / "label2id.json").open("w", encoding="utf-8") as f:
            json.dump(label2id, f, ensure_ascii=False, indent=2)
        with (d / "id2label.json").open("w", encoding="utf-8") as f:
            json.dump(id2label, f, ensure_ascii=False, indent=2)

    print(f"Saved label mappings ({len(label2id)} classes) to target directories.")

if __name__ == "__main__":
    main()
