"""Train the binary fruit-presence model using the fruit manifest and COCO128."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras

from config import IMAGE_SIZE, MODELS, RESULTS
from src.preprocess import make_dataset, read_labels, resize_rgb


def coco_negatives(directory):
    fruit_ids = {46, 47, 49}  # banana, apple, orange in COCO128's YOLO labels
    for path in sorted((directory / "images" / "train2017").glob("*.jpg")):
        annotation = directory / "labels" / "train2017" / path.with_suffix(".txt").name
        if not annotation.is_file():
            continue
        boxes = [line.split() for line in annotation.read_text().splitlines() if line.strip()]
        if boxes and fruit_ids.isdisjoint(int(box[0]) for box in boxes):
            yield path, boxes


def scene_and_crops(path, boxes):
    raw = tf.io.read_file(str(path))
    yield resize_rgb(raw)
    image = tf.io.decode_jpeg(raw, channels=3)
    for box in sorted(boxes, key=lambda box: float(box[3]) * float(box[4]), reverse=True)[:2]:
        x, y, width, height = map(float, box[1:])
        if min(width, height) < .10:
            continue
        bounds = [max(0, y - height / 2), max(0, x - width / 2),
                  min(1, y + height / 2), min(1, x + width / 2)]
        yield tf.image.crop_and_resize(image[None], [bounds], [0], IMAGE_SIZE)[0]


def embed(images, feature_model):
    features = []
    batch = []
    for image in images:
        batch.append(image)
        if len(batch) == 24:
            features.append(feature_model(tf.stack(batch), training=False).numpy())
            batch.clear()
    if batch:
        features.append(feature_model(tf.stack(batch), training=False).numpy())
    return np.concatenate(features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coco128", type=Path, help="Extracted COCO128 directory")
    parser.add_argument("--weights", default="imagenet", help="MobileNetV2 ImageNet weights or local H5 path")
    parser.add_argument("--manifest", type=Path, default=RESULTS / "labels.csv")
    parser.add_argument("--output", type=Path, default=MODELS / "fruit_presence.keras")
    args = parser.parse_args()

    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    keras.utils.set_random_seed(42)
    random = np.random.default_rng(42)
    splits, classes = read_labels(args.manifest)
    negatives = list(coco_negatives(args.coco128))
    if len(negatives) < 20:
        parser.error("COCO128 must contain at least 20 annotated scenes without COCO fruit labels.")
    random.shuffle(negatives)
    held_out = len(negatives) // 5

    backbone = keras.applications.MobileNetV2(
        input_shape=(160, 160, 3), alpha=.75, include_top=False, weights=args.weights, pooling="avg",
    )
    backbone.trainable = False
    inputs = keras.Input((*IMAGE_SIZE, 3))
    resized = keras.layers.Resizing(160, 160)(inputs)
    scaled = keras.layers.Rescaling(1 / 127.5, offset=-1)(resized)
    features = backbone(scaled, training=False)
    feature_model = keras.Model(inputs, features)

    def fruit_embeddings(split):
        batches = make_dataset(splits[split], classes, batch=24)
        return np.concatenate([feature_model(images, training=False).numpy() for images, _ in batches])

    def background_images(scenes):
        for path, boxes in scenes:
            yield from scene_and_crops(path, boxes)

    def training_backgrounds():
        yield from background_images(negatives[held_out:])
        for rgb in random.uniform(0, 255, size=(32, 3)):
            yield tf.broadcast_to(tf.cast(rgb, tf.float32), (*IMAGE_SIZE, 3))

    fruit_features = fruit_embeddings("train")
    background_features = embed(training_backgrounds(), feature_model)
    classifier = LogisticRegression(class_weight="balanced", C=2, max_iter=500)
    classifier.fit(np.concatenate([fruit_features, background_features]),
                   np.repeat([1, 0], [len(fruit_features), len(background_features)]))

    score = keras.layers.Dense(1, activation="sigmoid", name="fruit_score")
    model = keras.Model(inputs, score(features))
    score.set_weights([classifier.coef_.T, classifier.intercept_])

    fruit_scores = classifier.predict_proba(fruit_embeddings("test"))[:, 1]
    background_scores = classifier.predict_proba(embed(background_images(negatives[:held_out]), feature_model))[:, 1]
    for threshold in (.25, .35, .50):
        print(f"{threshold:.2f}: accepted fruit {(fruit_scores >= threshold).sum()}/{len(fruit_scores)}, "
              f"rejected backgrounds {(background_scores < threshold).sum()}/{len(background_scores)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)


if __name__ == "__main__":
    main()
