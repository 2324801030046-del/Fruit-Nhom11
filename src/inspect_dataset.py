"""Audit the current training manifest and summarize the separate splits."""
import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from config import DATASET, RESULTS
from src.make_fruit_folder_manifest import IMAGE_EXTENSIONS, LEGACY_FOLDERS


def inspect_dataset(manifest, output):
    with manifest.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    partitions = {row["path"]: row["split"] for row in rows}
    if len(partitions) != len(rows):
        raise ValueError("Manifest chứa đường dẫn trùng.")
    source = sorted(path for folder in DATASET.iterdir()
                    if folder.is_dir() and folder.name.casefold() not in LEGACY_FOLDERS
                    for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    formats, dimensions = Counter(), Counter()
    hashes = defaultdict(list)
    errors = []
    for path in source:
        relative = path.relative_to(DATASET).as_posix()
        try:
            raw_hash = sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as image:
                size, image_format = image.size, image.format
                image.verify()
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            errors.append({"path": relative, "error": str(error)})
            continue
        hashes[raw_hash].append(relative)
        formats[image_format] += 1
        dimensions[f"{size[0]}x{size[1]}"] += 1
    scanned = {path for matches in hashes.values() for path in matches}
    if missing := partitions.keys() - scanned:
        raise ValueError(f"Ảnh trong manifest bị thiếu hoặc không đọc được: {sorted(missing)}")
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    overlaps = [paths for paths in duplicates if sum(path in partitions for path in paths) > 1]
    if overlaps:
        raise ValueError(f"Ảnh trùng nội dung trong manifest: {overlaps}")

    splits = ("train", "val", "test")
    counts = {split: Counter(row["label"] for row in rows if row["split"] == split) for split in splits}
    classes = sorted({row["label"] for row in rows})
    audit = {
        "dataset": DATASET.as_posix(), "source_images": len(source), "valid_source_images": len(scanned),
        "manifest_images": len(rows), "by_split_and_label": counts, "formats": formats,
        "dimensions": dimensions, "identical_file_groups": duplicates,
        "duplicates_removed": sum(len(paths) - 1 for paths in duplicates),
        "unlisted_images": len(scanned - partitions.keys()), "unreadable": errors,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "class_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("label", *splits, "total"))
        for label in classes:
            sizes = [counts[split][label] for split in splits]
            writer.writerow((label, *sizes, sum(sizes)))
    with (output / "classes_missing_test.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("label", "train"))
        writer.writerows((label, counts["train"][label]) for label in classes if not counts["test"][label])
    with (output / "labels_pending.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("path",))
        writer.writerows((path,) for paths in hashes.values() if partitions.keys().isdisjoint(paths) for path in paths)

    columns, width, height = 4, 172, 154
    examples = [next(row for row in rows if row["label"] == label and row["split"] == "train") for label in classes]
    sheet = Image.new("RGB", (columns * width, (len(examples) + columns - 1) // columns * height), "#edf3e9")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("segoeui.ttf", 16)
    for i, row in enumerate(examples):
        x, y = i % columns * width, i // columns * height
        with Image.open(DATASET / row["path"]) as image:
            thumb = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (width-12, height-28))
        sheet.paste(thumb, (x + (width-thumb.width)//2, y + (height-28-thumb.height)//2))
        draw.text((x+8, y+height-25), row["label"], fill="#183828", font=font)
    sheet.save(output / "dataset_samples.jpg", quality=90)
    print(f"{len(source)} ảnh nguồn, {len(rows)} ảnh trong manifest, {len(classes)} nhãn.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=RESULTS / "labels.csv")
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    inspect_dataset(args.manifest, args.output_dir)
