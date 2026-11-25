"""Build train, validation and test lists from the named fruit folders."""
import argparse
import csv
from collections import Counter
from hashlib import sha256
from pathlib import Path
import re

from config import DATASET, RESULTS


FOLDER_NAMES = {
    "buoi": "Bưởi", "cam": "Cam", "chuoi": "Chuối", "duahau": "Dưa hấu",
    "le": "Lê", "mit": "Mít", "nho": "Nho", "quyt": "Quýt", "tao": "Táo",
    "thom": "Thơm", "xoai": "Xoài",
}
OUTPUT = RESULTS / "mobilenet_folder_fruits" / "manifest.csv"
LEGACY_FOLDERS = {"train", "test", "val"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def frame_order(path):
    parts = re.split(r"(\d+)", path.as_posix().casefold())
    return tuple(int(part) if i % 2 else part for i, part in enumerate(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, help="Giữ phân tập hiện có và chỉ thêm thư mục mới.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    rows = []
    if args.base_manifest:
        with args.base_manifest.open(encoding="utf-8-sig", newline="") as file:
            rows.extend((row["path"], row["label"], row["split"], row["group"]) for row in csv.DictReader(file))
    existing_folders = {Path(row[0]).parts[0].casefold() for row in rows}
    excluded_folders = LEGACY_FOLDERS | existing_folders
    folders = sorted(path for path in DATASET.iterdir() if path.is_dir() and path.name.casefold() not in excluded_folders)
    if not rows and len(folders) < 2:
        raise ValueError("Cần ít nhất hai thư mục tên trái cây.")
    if not folders:
        raise ValueError("Không tìm thấy thư mục trái cây mới.")

    fingerprints = {sha256((DATASET / path).read_bytes()).digest(): DATASET / path for path, _, _, _ in rows}
    used_labels = {row[1] for row in rows}
    duplicates = 0
    for folder in folders:
        label = FOLDER_NAMES.get(folder.name.casefold(), folder.name.replace("_", " ").capitalize())
        if label in used_labels:
            raise ValueError(f"Tên thư mục tạo nhãn bị trùng: {label}")
        used_labels.add(label)
        images = sorted((path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
                        key=frame_order)
        unique_images = []
        for path in images:
            digest = sha256(path.read_bytes()).digest()
            if digest in fingerprints:
                original = fingerprints[digest]
                if not original.is_relative_to(folder):
                    raise ValueError(f"Ảnh trùng ở hai nhãn: {original} và {path}")
                duplicates += 1
                continue
            fingerprints[digest] = path
            unique_images.append(path)

        count = len(unique_images)
        if count < 7:
            raise ValueError(f"Cần ít nhất 7 ảnh riêng biệt trong thư mục {folder.name}.")
        train_end = count * 70 // 100
        val_end = count * 85 // 100
        partitions = {"train": unique_images[:train_end], "val": unique_images[train_end:val_end],
                      "test": unique_images[val_end:]}
        for split, paths in partitions.items():
            rows.extend((path.relative_to(DATASET).as_posix(), label, split, f"{folder.name}:{split}") for path in paths)
        print(label, {split: len(paths) for split, paths in partitions.items()})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("path", "label", "split", "group"))
        writer.writerows(rows)
    print("Tổng:", dict(Counter(row[2] for row in rows)), "ảnh trùng đã bỏ:", duplicates)


if __name__ == "__main__":
    main()
