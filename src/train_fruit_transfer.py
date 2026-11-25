"""Fit a fruit classifier on the frozen MobileNetV2 image features."""
import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import tensorflow as tf
from tensorflow import keras

from config import MODELS, RESULTS
from src.evaluate import evaluate
from src.predict import load_presence_classifier
from src.preprocess import make_dataset, read_labels


def encode(dataset, encoder, augment=None):
    features, labels = [], []
    for images, batch_labels in dataset:
        features.append(encoder(images, training=False).numpy())
        labels.append(batch_labels.numpy())
        if augment is not None:
            for _ in range(2):
                features.append(encoder(augment(images, training=True), training=False).numpy())
                labels.append(batch_labels.numpy())
    return np.concatenate(features), np.concatenate(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "mobilenet_transfer")
    parser.add_argument("--manifest", type=Path, default=RESULTS / "labels.csv")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    keras.utils.set_random_seed(42)
    start = monotonic()
    splits, classes = read_labels(args.manifest)
    presence = load_presence_classifier(MODELS)
    encoder = keras.Model(presence.inputs, presence.get_layer("fruit_score").input)
    encoder.trainable = False

    augment = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(.08, fill_mode="constant", fill_value=230),
        keras.layers.RandomZoom((-.12, .22), fill_mode="constant", fill_value=230),
        keras.layers.RandomTranslation(.09, .09, fill_mode="constant", fill_value=230),
        keras.layers.RandomBrightness(.12, value_range=(0, 255)),
    ])
    batch_size = 24
    train = make_dataset(splits["train"], classes, batch_size)
    validation = make_dataset(splits["val"], classes, batch_size)
    train_features, train_labels = encode(train, encoder, augment)
    val_features, val_labels = encode(validation, encoder)

    candidates = []
    for regularization in (.03, .15, .75):
        candidate = LogisticRegression(C=regularization, class_weight="balanced", max_iter=500)
        candidate.fit(train_features, train_labels)
        scores = candidate.predict_proba(val_features)
        accuracy = float(np.mean(scores.argmax(axis=1) == val_labels))
        loss = log_loss(val_labels, scores, labels=candidate.classes_)
        candidates.append((accuracy, loss, candidate))
        print(f"C={regularization}: val_accuracy={accuracy:.4f}, val_loss={loss:.4f}", flush=True)
    accuracy, loss, classifier = max(candidates, key=lambda item: (item[0], -item[1]))

    head = keras.layers.Dense(len(classes), activation="softmax", name="fruit_name")
    model = keras.Model(encoder.inputs, head(encoder.output), name="fruit_mobilenet")
    head.set_weights([classifier.coef_.T, classifier.intercept_])
    checkpoint = output / "fruit_classifier.keras"
    model.save(checkpoint)
    (output / "class_names.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "validation.json").write_text(json.dumps({
        "accuracy": accuracy, "loss": loss, "C": classifier.C, "images": len(val_labels),
    }, indent=2), encoding="utf-8")
    evaluate(checkpoint, output, splits, classes, batch_size, monotonic() - start)


if __name__ == "__main__":
    main()
