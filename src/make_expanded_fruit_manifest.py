"""Add labeled images from val and test to the named-folder dataset."""
import csv
from collections import Counter
from pathlib import Path

from config import DATASET, LEGACY_RESULTS, RESULTS
from src.make_fruit_folder_manifest import IMAGE_EXTENSIONS, OUTPUT as FOLDER_MANIFEST


OUTPUT = RESULTS / "mobilenet_expanded_fruits" / "manifest.csv"
SOURCE_FOLDERS = ("val", "test")
LABEL_ALIASES = {"Nho xanh": "Nho", "Dứa": "Thơm"}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main():
    base = read_csv(FOLDER_MANIFEST)
    renames = {row["old_path"]: row["path"] for row in read_csv(OUTPUT.with_name("legacy_renames.csv"))}
    available = {
        path.relative_to(DATASET).as_posix()
        for folder in SOURCE_FOLDERS
        for path in (DATASET / folder).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if not available:
        raise ValueError("Không tìm thấy ảnh trong val và test.")

    legacy = []
    accounted_for = set()
    for row in read_csv(LEGACY_RESULTS / "labels.csv"):
        if Path(row["path"]).parts[0] not in SOURCE_FOLDERS:
            continue
        current = renames.get(row["path"], row["path"])
        if current not in available:
            continue
        if current in accounted_for:
            raise ValueError(f"Ảnh có nhiều nhãn trong manifest: {current}")
        accounted_for.add(current)
        if row["label"].startswith("Quả tròn "):
            continue
        legacy.append({**row, "path": current, "label": LABEL_ALIASES.get(row["label"], row["label"]),
                       "group": f"legacy:{row['group']}"})
    if unlabelled := available - accounted_for:
        raise ValueError(f"Chưa có nhãn cho: {', '.join(sorted(unlabelled))}")

    combined = base + legacy
    trained = {row["label"] for row in combined if row["split"] == "train"}
    if missing := {row["label"] for row in combined} - trained:
        raise ValueError(f"Nhãn chưa có ảnh train: {', '.join(sorted(missing))}")
    with OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("path", "label", "split", "group"))
        writer.writeheader()
        writer.writerows(combined)
    print("Ảnh từ val/test theo nhãn:", dict(Counter(row["label"] for row in legacy)))
    print("Phân tập:", dict(Counter(row["split"] for row in combined)))
    print("Tổng nhãn:", len(trained))


if __name__ == "__main__":
    main()
