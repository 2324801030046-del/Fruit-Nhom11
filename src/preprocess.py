"""Đọc danh sách nhãn và chia dữ liệu theo các nhóm ảnh đã rà soát."""
import csv
from collections import defaultdict
from io import BytesIO

import numpy as np
from PIL import Image
import tensorflow as tf

from config import DATASET, IMAGE_SIZE, RESULTS


def read_labels(manifest=RESULTS / "labels.csv"):
    splits = {split: [] for split in ("train", "val", "test")}
    seen_paths = set()
    group_splits = defaultdict(set)
    with manifest.open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            path = DATASET / row["path"]
            if not path.is_file() or path in seen_paths:
                raise ValueError(f"Đường dẫn thiếu hoặc trùng: {path}")
            seen_paths.add(path)
            group_splits[row["group"]].add(row["split"])
            splits[row["split"]].append((str(path), row["label"]))
    classes = sorted({label for _, label in splits["train"]})
    if len(classes) < 2 or any(
        not rows or not {label for _, label in rows}.issubset(classes)
        for rows in splits.values()
    ):
        raise ValueError("Cần ít nhất hai lớp train, và val/test phải chứa nhãn đã được học.")
    if any(len(partitions) != 1 for partitions in group_splits.values()):
        raise ValueError("Một nhóm ảnh xuất hiện trong nhiều tập.")
    return splits, classes


def decode_webp(raw):
    with Image.open(BytesIO(raw.numpy())) as image:
        return np.asarray(image.convert("RGB"))


def resize_rgb(raw):
    def decode_jpeg():
        dimensions = tf.io.extract_jpeg_shape(raw)
        # Native JPEG downsampling keeps large phone photos cheap to read.
        return tf.cond(tf.reduce_min(dimensions[:2]) >= 2048,
                       lambda: tf.io.decode_jpeg(raw, channels=3, ratio=4),
                       lambda: tf.io.decode_jpeg(raw, channels=3))

    def decode_other():
        is_webp = tf.cond(tf.strings.length(raw) >= 12,
                          lambda: tf.equal(tf.strings.substr(raw, 8, 4), "WEBP"),
                          lambda: False)
        return tf.cond(is_webp,
                       lambda: tf.py_function(decode_webp, [raw], Tout=tf.uint8),
                       lambda: tf.io.decode_image(raw, channels=3, expand_animations=False))

    image = tf.cond(tf.io.is_jpeg(raw), decode_jpeg, decode_other)
    image.set_shape((None, None, 3))
    return tf.image.resize(image, IMAGE_SIZE, antialias=True)


def make_dataset(rows, classes, batch, training=False, cache=False):
    paths, labels = zip(*rows)
    ds = tf.data.Dataset.from_tensor_slices((list(paths), [classes.index(label) for label in labels]))
    ds = ds.map(
        lambda path, label: (resize_rgb(tf.io.read_file(path)), label),
        num_parallel_calls=2,
    )
    if cache:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(len(rows), seed=42)
    return ds.batch(batch).prefetch(1)
