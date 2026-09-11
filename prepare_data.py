import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from config import CONFIG, RAW_DATA_DIR


REQUIRED_FIELDS = ("id", "label_id", "category", "text", "keywords")


def validate_record(record: dict) -> None:
    if not isinstance(record, dict) or any(not isinstance(record.get(k), str) for k in REQUIRED_FIELDS):
        raise ValueError("Expected string fields: id, label_id, category, text, keywords")
    if any(not record[k].strip() for k in ("id", "label_id", "category", "text")):
        raise ValueError("id, label_id, category and text must not be blank")


def parse_line(line: str) -> dict[str, str]:
    parts = line.rstrip("\r\n").split("_!_", 4)
    if len(parts) != 5:
        raise ValueError("Expected five '_!_'-separated fields")
    record = dict(zip(REQUIRED_FIELDS, parts))
    validate_record(record)
    return record


def validate_mapping(mapping: dict) -> None:
    if not mapping or any(not isinstance(k, str) or not k for k in mapping):
        raise ValueError("Label mapping requires nonempty string labels")
    if any(type(v) is not int for v in mapping.values()):
        raise ValueError("Label indices must be integers")
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("Label indices must be unique and continuous from zero")


def audit(splits: dict[str, list[dict]]) -> dict:
    # 查测试集
    report = {"splits": {}, "overlap": {}}
    for name, records in splits.items():
        report["splits"][name] = {
            "rows": len(records),
            "class_counts": dict(sorted(Counter(r["label_id"] for r in records).items())),
            "duplicate_id_rows": len(records) - len({r["id"] for r in records}),
            "duplicate_title_rows": len(records) - len({r["text"] for r in records}),
        }
    for first, second in combinations(splits, 2):
        titles = {r["text"] for r in splits[first]} & {r["text"] for r in splits[second]}
        report["overlap"][f"{first}-{second}"] = {
            "shared_ids": len({r["id"] for r in splits[first]} & {r["id"] for r in splits[second]}),
            "shared_titles": len(titles),
            "affected_second_rows": sum(r["text"] in titles for r in splits[second]),
        }
    return report


def prepare(raw_files: dict[str, Path], data_dir: Path) -> dict:

    splits = {}
    for split, path in raw_files.items():
        records = []
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    records.append(parse_line(line))
                except ValueError as error:
                    raise ValueError(f"{path}:{number}: {error}")
        if not records:
            raise ValueError(f"{path}: dataset is empty")
        splits[split] = records

    # 仅根据训练集建立全局统一的标签索引，确保训练与评测口径对齐
    labels = sorted({record["label_id"] for record in splits["train"]})
    mapping = {label: index for index, label in enumerate(labels)}
    validate_mapping(mapping)
    for split, records in splits.items():
        unknown = {record["label_id"] for record in records} - mapping.keys()
        if unknown:
            raise ValueError(f"{split}: labels absent from training: {sorted(unknown)}")

    audit_report = audit(splits)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
        (data_dir / f"{split}.jsonl").write_text(payload, encoding="utf-8")
    for name, content in {
        "label2id.json": mapping,
        "id2label.json": {index: label for label, index in mapping.items()},
        "audit.json": audit_report,
    }.items():
        (data_dir / name).write_text(
            json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return audit_report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare the canonical Toutiao splits and data audit.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--data-dir", type=Path, default=CONFIG.data_dir)
    args = parser.parse_args(argv)
    raw_files = {
        "train": args.raw_dir / "train_3k.txt",
        "dev": args.raw_dir / "dev_1k.txt",
        "test": args.raw_dir / "test_1k.txt",
    }
    audit_report = prepare(raw_files, args.data_dir)
    print(json.dumps(audit_report, ensure_ascii=False, indent=2))
    return audit_report


if __name__ == "__main__":
    main()

