"""Xuất nhãn được đọc từ ảnh và nhóm quả theo đặc điểm nhìn thấy."""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Mỗi hàng là một nhóm ảnh có thể cùng quả. Các nhóm chắc chắn thấp được bỏ qua.
GROUPS = [
    ("Táo", "train", [0], [], [721, 829]),
    ("Táo", "train", [3, 7], [], []),
    ("Táo", "train", [17], [], [453, 823]),
    ("Táo", "train", [67], [239, 298, 313, 442, 722], []),
    ("Táo", "train", [35], [26, 88, 90, 194, 225, 272, 296, 335, 391, 408, 474, 535, 542, 683, 745, 799, 857, 916], []),
    ("Táo", "val", [42], [18, 67, 71, 87, 675, 803], []),
    ("Táo", "test", [61], [], []),
    ("Lê", "train", [19, 0, 52], [31, 70, 115, 172, 235, 246, 572, 691, 721, 829, 836, 939, 316, 558, 923, 954], []),
    ("Lê", "train", [1, 55, 53], [65, 91, 168, 380, 496, 504, 527, 621, 650, 699, 720, 770, 798, 937], []),
    ("Lê", "val", [18, 45], [201, 329, 375, 413, 563, 568, 585, 634, 635, 638, 728, 757, 820, 864, 904], []),
    ("Lê", "test", [65, 15], [157, 242, 315, 526, 570, 644, 737, 775, 925, 930, 946], []),
    ("Cam", "train", [8, 29], [], [789]),
    ("Cam", "train", [2], [], [6, 8, 102, 141, 253, 271, 678]),
    ("Cam", "train", [6], [22, 37, 60, 113, 114, 193, 205, 210, 230, 254, 508, 545, 569, 574, 790, 792, 830, 924, 943], []),
    ("Cam", "val", [50], [], []),
    ("Cam", "test", [27], [21, 140, 146, 273, 297, 310, 346, 365, 571, 593, 604, 628, 651, 712, 727, 774, 778, 852, 905, 942], []),
    ("Dâu tây", "train", [17, 59, 67], [29, 32, 237, 250, 251, 330, 453, 463, 633, 823, 846, 957], []),
    ("Dâu tây", "val", [70], [], []),
    ("Dâu tây", "test", [68], [], []),
    ("Bơ", "train", [13, 36], [], []),
    ("Bơ", "val", [26], [], []),
    ("Bơ", "test", [49], [], []),
    ("Chuối", "train", [4, 37, 64], [80, 318, 285, 426, 534, 598, 715, 822, 856, 132, 498, 695, 713], []),
    ("Chuối", "test", [9, 28, 33, 38], [10, 23, 52, 53, 127, 248, 348, 523, 595, 606, 690, 748, 881, 915], []),
    ("Vải", "train", [22], [], []),
    ("Mâm xôi", "train", [5], [], []),
    ("Việt quất", "train", [14], [], []),
    ("Dừa", "train", [41], [], []),
    ("Dứa", "train", [16, 47], [], []),
    ("Kiwi", "train", [15, 66], [69, 110, 139, 243, 291, 347, 379, 382, 422, 456, 507, 524, 814], []),
    ("Thanh long", "train", [9, 25], [], [501, 523, 768, 848, 866, 912]),
    ("Khế", "train", [2, 60], [6, 8, 102, 141, 253, 271, 678, 75, 211, 241, 281, 331, 334, 356, 444, 492, 573, 579, 667, 679, 959], []),
    ("Dưa pepino", "train", [1, 55, 71], [40, 57, 111, 129, 134, 186, 218, 234, 278, 284, 327, 340, 369, 404, 529, 544, 557, 652, 873, 893], []),
    ("Chà là", "train", [12, 20], [], []),
    ("Nho xanh", "train", [24, 44, 53], [], [380, 496, 650, 720, 770]),
    ("Mận tím", "train", [32], [], []),
    ("Đào", "train", [62, 63, 67, 29], [30, 76, 190, 322, 421, 625, 672, 819, 845, 859, 877, 58, 255, 516, 540, 744, 749, 771, 872, 882, 952, 389, 397, 432, 789], []),
    ("Đào", "test", [52, 66, 59], [25, 33, 66, 89, 130, 195, 199, 275, 495, 670, 688, 709, 825, 886], []),
    ("Mơ", "train", [27, 63], [15, 35, 123, 204, 307, 434, 450, 894, 910, 921], []),
    ("Chanh vàng", "train", [10], [49, 349, 406, 736], []),
    ("Hồng", "train", [6], [86, 108, 187, 244, 464, 869, 911], []),
    ("Hồng", "test", [46], [7, 751, 889], []),
    ("Salak", "train", [28, 58], [], [23, 127]),
    ("Cà chua", "train", [48], [665, 919], []),
    ("Quất", "train", [40, 56], [], [256, 416, 718, 723]),
    ("Chanh dây tím", "train", [31], [], []),
    ("Anh đào", "train", [11], [], []),
    ("Bưởi", "train", [23], [], []),
    ("Ổi", "train", [4], [], [80, 318]),
    ("Cam", "train", [6, 48], [182, 223, 553, 616, 623, 99, 931], []),
    ("Quýt", "train", [48], [], [99, 665, 919, 931]),
    ("Lê", "test", [35], [63, 897], []),
    ("Lê", "train", [1, 15], [364, 402, 687, 945], []),
    ("Đào", "train", [21], [51, 170, 584, 589, 653, 918], []),
    ("Chanh vàng", "test", [10], [], [49, 349, 406, 736, 610, 700, 884]),
    ("Mơ", "train", [56], [416, 718, 723], []),
    ("Xoài", "train", [39, 57], [], [13]),
    ("Mộc qua", "train", [30, 54], [117, 231, 552, 619, 674, 68, 73, 149, 196, 352, 441, 549, 550, 734, 827, 926], []),
    ("Tamarillo", "train", [9, 43], [], [332, 374, 485, 518, 523, 694, 831, 896, 960]),
    ("Sapôchê", "train", [34, 51, 46], [], [7, 751, 889]),
    ("Quả xương rồng", "train", [1, 15, 39, 45, 69], [13, 28, 59, 154, 174, 419, 427, 469, 490, 530, 566, 627, 654, 838, 885, 901, 920, 944], []),
    ("Mận tím", "test", [59], [104, 200, 236, 342, 448, 821], []),
    # Nhãn mô tả ngoại hình cho các nhóm chưa xác định chắc tên quả.
    ("Quả tròn cam đốm nâu", "train", [21, 56], [189, 537, 256], []),
    ("Quả tròn vàng xanh vỏ trơn", "train", [10, 30], [14, 610, 700, 884], []),
    ("Quả tròn vàng vỏ sần", "train", [30], [], [14, 117, 231, 552, 619, 674]),
    ("Táo", "train", [42], [287, 338, 395, 465, 511, 522, 719, 763, 818, 851, 929, 935, 961], []),
    ("Lê", "train", [42], [48, 502, 546, 615], []),
]


def export_labels():
    output = ROOT / "results/legacy_root"
    output.mkdir(exist_ok=True)
    with (ROOT / "results/label_review/index.csv").open(encoding="utf-8") as file:
        reviewed = list(csv.DictReader(file))
    labels = {}
    for group, (label, split, clusters, include, exclude) in enumerate(GROUPS):
        matched = set()
        for row in reviewed:
            image_id = int(row["id"])
            if int(row["cluster"]) not in clusters or image_id in exclude:
                continue
            if include and image_id not in include:
                continue
            if row["path"] in labels:
                raise ValueError(f"Nhãn trùng cho ảnh {image_id}")
            labels[row["path"]] = {"path": row["path"], "label": label, "split": split, "group": group}
            matched.add(image_id)
        if include and matched != set(include):
            raise ValueError(f"Thiếu ảnh khi gắn nhãn nhóm {group}: {set(include) - matched}")
    corrections = ROOT / "results/review_wrong/anh_du_doan_sai.csv"
    if corrections.is_file():
        with corrections.open(encoding="utf-8-sig", newline="") as review_file:
            for row in csv.DictReader(review_file):
                confirmed = row["Nhan ban xac nhan"].strip()
                if confirmed:
                    path = Path(row["Anh"]).relative_to("dataset/anhtraicay/data").as_posix()
                    labels[path]["label"] = "Chuối" if confirmed == "Chuối đỏ Dacca" else confirmed
    with (ROOT / "results/label_review/user_filenames.csv").open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            original = labels[row["old_path"]]
            if not row["path"]:
                del labels[row["old_path"]]
                continue
            original.update(path=row["path"], label=row["label"])
    with (output / "labels.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "label", "split", "group"])
        writer.writeheader()
        writer.writerows(labels.values())
    with (output / "labels_pending.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "cluster", "path"])
        writer.writeheader()
        dataset = ROOT / "dataset/anhtraicay/data"
        writer.writerows(row for row in reviewed if row["path"] not in labels and (dataset / row["path"]).is_file())
    counts = Counter((row["label"], row["split"]) for row in labels.values())
    classes = sorted({row["label"] for row in labels.values()})
    with (output / "class_summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(("label", "train", "val", "test", "total"))
        for label in classes:
            sizes = [counts[label, split] for split in ("train", "val", "test")]
            writer.writerow((label, *sizes, sum(sizes)))
    with (output / "classes_missing_test.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        writer.writerow(("label", "train"))
        writer.writerows((label, counts[label, "train"]) for label in classes if not counts[label, "test"])
    print(f"{len(labels)} ảnh đã gắn nhãn.")


if __name__ == "__main__":
    export_labels()
