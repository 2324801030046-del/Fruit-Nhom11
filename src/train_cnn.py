"""Huấn luyện CNN và khôi phục checkpoint có validation loss thấp nhất."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import tensorflow as tf
from tensorflow.keras import layers

from config import IMAGE_SIZE, RESULTS
from src.evaluate import evaluate
from src.preprocess import make_dataset, read_labels


def build_model(class_count):
    return tf.keras.Sequential([
        layers.Input((*IMAGE_SIZE, 3)),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.10, fill_mode="reflect"),
        layers.RandomBrightness(0.08, value_range=(0, 255)),
        layers.RandomContrast(0.12),
        layers.RandomZoom(0.10),
        layers.Rescaling(1 / 255),
        layers.Conv2D(24, 5, strides=3, padding="same", use_bias=False),
        layers.BatchNormalization(momentum=0.85),
        layers.ReLU(),
        layers.MaxPooling2D(),
        layers.Conv2D(48, 3, padding="same", use_bias=False),
        layers.BatchNormalization(momentum=0.85),
        layers.ReLU(),
        layers.MaxPooling2D(),
        layers.Conv2D(96, 3, padding="same", use_bias=False),
        layers.BatchNormalization(momentum=0.85),
        layers.ReLU(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", use_bias=False),
        layers.BatchNormalization(momentum=0.85),
        layers.ReLU(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(96, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(class_count, activation="softmax"),
    ], name="fruit_cnn")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--quick", action="store_true", help="Chạy thử một epoch trong results/quick.")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--manifest", type=Path, default=RESULTS / "labels.csv")
    parser.add_argument("--output-dir", type=Path, help="Lưu mô hình và kết quả cùng nhau cho lượt thử riêng.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs và --batch-size phải lớn hơn 0")

    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(42)
    splits, classes = read_labels(args.manifest)
    output = args.output_dir or (RESULTS / "quick" if args.quick else RESULTS / "cnn_experiment")
    model_dir = output
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "fruit_classifier.keras"
    train = make_dataset(splits["train"], classes, args.batch_size, training=True, cache=args.cache)
    val = make_dataset(splits["val"], classes, args.batch_size, cache=args.cache)
    model = build_model(len(classes))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    counts = Counter(label for _, label in splits["train"])
    class_weight = {index: math.sqrt(len(splits["train"]) / (len(classes) * counts[label]))
                    for index, label in enumerate(classes)}
    start = time.monotonic()
    model.fit(train, validation_data=val, epochs=1 if args.quick else args.epochs,
              verbose=2, shuffle=False, class_weight=class_weight, callbacks=[
        tf.keras.callbacks.CSVLogger(str(output / "history.csv")),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True),
    ])
    elapsed = time.monotonic() - start
    (model_dir / "class_names.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    (model_dir / "preprocessing.json").write_text(json.dumps({
        "input_size": list(IMAGE_SIZE), "colors": "RGB", "resize": "bilinear antialias",
        "pixel_range": [0, 255], "scale_in_model": 1 / 255,
    }, indent=2), encoding="utf-8")
    if not args.quick:
        evaluate(checkpoint, output, splits, classes, args.batch_size, elapsed)


if __name__ == "__main__":
    main()
