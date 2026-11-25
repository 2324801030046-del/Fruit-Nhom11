import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def make_sheets():
    root = Path(__file__).resolve().parents[1]
    dataset = root / "dataset" / "anhtraicay" / "data"
    output = root / "results" / "label_review"
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(dataset.glob("*/*/*.jpg"))
    features = []
    tiles = []
    for path in paths:
        with Image.open(path) as image:
            tile = ImageOps.contain(image.convert("RGB"), (72, 72))
        tiles.append(tile)
        small = np.asarray(tile.resize((24, 24)), dtype=np.float32) / 255
        hsv = np.asarray(tile.convert("HSV"))
        mask = (hsv[:, :, 1] > 35) | (hsv[:, :, 2] < 170)
        pixels = hsv[mask]
        hist, _ = np.histogramdd(pixels, bins=(12, 4, 4), range=((0, 256),) * 3)
        histogram = normalize(np.sqrt(hist).reshape(1, -1))[0]
        thumbnail = normalize((1 - small).reshape(1, -1))[0]
        features.append(np.r_[histogram, thumbnail * 1.35])
    clusters = KMeans(n_clusters=72, random_state=42, n_init=5).fit_predict(features)
    with (output / "index.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("id", "cluster", "path"))
        writer.writerows((i, c, p.relative_to(dataset).as_posix()) for i, (c, p) in enumerate(zip(clusters, paths)))
    groups = [np.flatnonzero(clusters == c) for c in range(72)]
    for page in range(18):
        panel = groups[page * 4:page * 4 + 4]
        sheet = Image.new("RGB", (960, sum(26 + 92 * ((len(g) + 11) // 12) for g in panel)), "#eaeaea")
        draw = ImageDraw.Draw(sheet)
        y = 0
        for c, group in enumerate(panel, start=page * 4):
            draw.text((4, y + 4), f"Cluster {c}", fill="#111")
            y += 26
            for j, i in enumerate(group):
                x, offset = (j % 12) * 80, (j // 12) * 92
                sheet.paste(tiles[i], (x + 4, y + offset))
                draw.text((x + 8, y + offset + 74), str(i), fill="#111")
            y += 92 * ((len(group) + 11) // 12)
        sheet.save(output / f"{page:02}.jpg", quality=93)


if __name__ == "__main__":
    make_sheets()
