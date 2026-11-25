"""Dự đoán từ ảnh tải lên hoặc ảnh trên đĩa."""
import argparse
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import tensorflow as tf

from config import CONFIDENCE_THRESHOLD, FRUIT_PRESENCE_THRESHOLD, IMAGE_SIZE, MAX_IMAGE_BYTES, MODELS, WEB_MODEL_DIR
from src.preprocess import resize_rgb


def load_classifier(directory=WEB_MODEL_DIR):
    names = json.loads((directory / "class_names.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(directory / "fruit_classifier.keras", compile=False)
    if model.input_shape[1:] != (*IMAGE_SIZE, 3):
        raise ValueError("Checkpoint không nhận ảnh RGB 224 × 224 như pipeline hiện tại.")
    if len(names) != model.output_shape[-1]:
        raise ValueError("Số nhãn khác kích thước đầu ra mô hình.")
    return model, names


def load_presence_classifier(directory=WEB_MODEL_DIR):
    model_path = directory / "fruit_presence.keras"
    if not model_path.is_file():
        model_path = MODELS / "fruit_presence.keras"
    return tf.keras.models.load_model(model_path, compile=False)


class PhotoTooLarge(ValueError):
    pass


def predict_image(file, classifier, presence_classifier=None):
    raw = file.read(MAX_IMAGE_BYTES + 1)
    if len(raw) > MAX_IMAGE_BYTES:
        raise PhotoTooLarge("Ảnh vượt quá 5 MB.")
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Chỉ hỗ trợ ảnh JPG, JPEG, PNG và WebP.")
            image.verify()
        tensor = resize_rgb(raw)
    except (OSError, tf.errors.OpError, Image.DecompressionBombError) as error:
        raise ValueError("Không đọc được ảnh. Hãy chọn một ảnh JPG, PNG hoặc WebP hợp lệ.") from error
    batch = tensor[None, ...]
    if presence_classifier is not None:
        fruit_score = float(presence_classifier(batch, training=False).numpy()[0, 0])
        if fruit_score < FRUIT_PRESENCE_THRESHOLD:
            return {"non_fruit": True, "certain": False, "top3": []}

    model, names = classifier
    scores = model(batch, training=False).numpy()[0]
    best = scores.argsort()[::-1][:3]
    return {
        "non_fruit": False,
        "certain": bool(scores[best[0]] >= CONFIDENCE_THRESHOLD),
        "top3": [{"label": names[index], "probability": float(scores[index])} for index in best],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    try:
        with args.image.open("rb") as file:
            result = predict_image(file, load_classifier(), load_presence_classifier())
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
