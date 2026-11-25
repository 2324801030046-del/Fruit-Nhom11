"""Tạo tập kiểm tra theo ảnh cho 29 nhãn thiếu test độc lập."""
from collections import defaultdict
import csv
from pathlib import Path
import random

results = Path("results")
output = results / "test_29"
output.mkdir(exist_ok=True)
with (results / "legacy_root/labels.csv").open(encoding="utf-8", newline="") as file:
    rows = list(csv.DictReader(file))
with (results / "legacy_root/classes_missing_test.csv").open(encoding="utf-8-sig", newline="") as file:
    labels = {row["label"] for row in csv.DictReader(file)}

candidates = defaultdict(list)
experiment = []
for row in rows:
    if row["split"] == "test":
        continue
    experiment.append(row)
    if row["split"] == "train" and row["label"] in labels:
        candidates[row["label"]].append(row)

rng = random.Random(42)
for label, images in sorted(candidates.items()):
    images.sort(key=lambda row: row["path"])
    for row in rng.sample(images, max(1, round(len(images) * 0.2))):
        row["split"] = "test"
        # Lượt kiểm tra này tách theo ảnh; giữ mã nhóm nguồn để thấy mức trùng nhóm.
        row["source_group"] = row["group"]
        row["group"] = f"image_holdout_{row['group']}"

for row in experiment:
    row.setdefault("source_group", row["group"])
assert {row["label"] for row in experiment if row["split"] == "test"} == labels
assert labels.issubset(row["label"] for row in experiment if row["split"] == "train")
with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["path", "label", "split", "group", "source_group"])
    writer.writeheader()
    writer.writerows(experiment)
print(f"Labels: {len(labels)}; test images: {sum(row['split'] == 'test' for row in experiment)}")
