"""Đánh giá mô hình đã lưu và xuất biểu đồ từ số liệu thực tế."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from config import WEB_MODEL_DIR, WEB_RESULTS_DIR
from src.preprocess import make_dataset, read_labels


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def evaluate(checkpoint, output, splits, classes, batch=32, elapsed=None):
    model = tf.keras.models.load_model(checkpoint, compile=False)
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    test = make_dataset(splits["test"], classes, batch)
    scores = model.evaluate(test, return_dict=True, verbose=0)
    predictions = model.predict(test, verbose=0)
    actual = [classes.index(label) for _, label in splits["test"]]
    guessed = predictions.argmax(axis=1)
    report = classification_report(actual, guessed, labels=list(range(len(classes))),
                                   target_names=classes, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(output / "classification_report.csv", encoding="utf-8-sig")
    summary = dict(scores)
    tested = [name for name in classes if report[name]["support"]]
    macro = {key: float(np.mean([report[name][key] for name in tested]))
             for key in ("precision", "recall", "f1-score")}
    summary.update({"macro": macro, "weighted": report["weighted avg"],
                    "tested_classes": tested,
                    "classes_without_test_images": [name for name in classes if name not in tested],
                    "test_images": len(actual), "model_bytes": checkpoint.stat().st_size})
    if elapsed is not None:
        summary["training_seconds"] = round(elapsed, 2)

    history_path = output / "history.csv"
    if history_path.is_file():
        history = pd.read_csv(history_path)
        best = history.loc[history["val_loss"].idxmin()]
        summary["best_checkpoint"] = {key: float(best[key]) for key in ("loss", "accuracy", "val_loss", "val_accuracy")}
        summary["best_checkpoint"]["epoch"] = int(best["epoch"]) + 1
        for metric, ylabel in (("accuracy", "Accuracy"), ("loss", "Loss")):
            history.plot(x="epoch", y=[metric, f"val_{metric}"], marker="o", xlabel="Epoch (từ 0)", ylabel=ylabel)
            save_figure(output / f"{metric}.png")
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    _, axis = plt.subplots(figsize=(max(7, len(classes) * 0.46), max(5, len(tested) * 0.55)))
    ConfusionMatrixDisplay.from_predictions(actual, guessed, labels=list(range(len(classes))),
                                            display_labels=classes, cmap="Blues",
                                            xticks_rotation=55, ax=axis)
    axis.set_ylim(len(classes) - 0.5, -0.5)
    save_figure(output / "confusion_matrix.png")
    pd.DataFrame({split: pd.Series([label for _, label in rows]).value_counts()
                  for split, rows in splits.items()}).reindex(classes).plot.bar(
                      rot=55, ylabel="Số ảnh", figsize=(12, 5))
    save_figure(output / "class_counts.png")
    pd.DataFrame(report).T.loc[tested, ["precision", "recall", "f1-score"]].plot.bar(rot=0, ylim=(0, 1))
    save_figure(output / "class_metrics.png")

    pd.DataFrame({"path": [path for path, _ in splits["test"]], "actual": [classes[i] for i in actual],
                  "predicted": [classes[i] for i in guessed], "confidence": np.max(predictions, axis=1)
                  }).to_csv(output / "test_predictions.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    manifest = WEB_RESULTS_DIR / "manifest.csv"
    splits, classes = read_labels(manifest) if manifest.is_file() else read_labels()
    saved = json.loads((WEB_MODEL_DIR / "class_names.json").read_text(encoding="utf-8"))
    if classes != saved:
        raise ValueError("Thứ tự nhãn hiện tại khác mô hình đã lưu.")
    previous = WEB_RESULTS_DIR / "metrics.json"
    elapsed = json.loads(previous.read_text(encoding="utf-8")).get("training_seconds") if previous.is_file() else None
    evaluate(WEB_MODEL_DIR / "fruit_classifier.keras", WEB_RESULTS_DIR, splits, classes, elapsed=elapsed)
