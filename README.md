# Nhóm 11 – Phân loại trái cây CNN

Website Flask dùng hai mô hình MobileNetV2 để kiểm tra ảnh có trái cây và dự đoán tên quả. Trang **Giới thiệu** liệt kê các nhãn của checkpoint đang chạy.

## Dữ liệu và kết quả hiện tại

Bộ dữ liệu hiện có 11 thư mục loại quả trong `dataset/anhtraicay/data`. Sau khi lọc 53 ảnh trùng nội dung, manifest `results/labels.csv` có **2.557 ảnh** gồm **1.831 train, 358 validation và 368 test**. Validation và test là các phần giữ lại từ 11 thư mục hiện có. Ba nhãn Dâu tây, Dưa pepino và Salak thuộc lần huấn luyện trước.

- `models/fruit_classifier.keras`, `models/fruit_presence.keras`, `models/class_names.json` và `models/preprocessing.json` là bộ mô hình mặc định.
- `results/class_summary.csv` và `results/dataset_audit.json` ghi số ảnh theo nhãn, định dạng và các nhóm ảnh trùng.
- `results/metrics.json`, `results/classification_report.csv` và `results/test_predictions.csv` ghi kết quả bộ phân loại. `results/website_test_summary.csv` ghi kết quả sau khi áp dụng cả bộ lọc trái cây.
- `REPORT.md` tổng hợp phương pháp, kết quả và giới hạn của bộ test 11 nhãn hiện tại.

## Chạy website bằng PowerShell

```powershell
# Chạy từ thư mục gốc của repo
.\.venv\Scripts\python.exe app.py
```

Nếu chưa có môi trường `.venv`, chạy `python -m venv .venv`, sau đó chạy `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

Mở [website Nhóm 11](http://127.0.0.1:5000/) và đăng nhập mặc định bằng `admin` / `admin`. Có thể đặt `FRUIT_ADMIN_PASSWORD` để đổi mật khẩu và `FRUIT_SESSION_KEY` để giữ phiên qua lần khởi động lại Flask.

- **Tải ảnh lên** hỗ trợ JPG, JPEG, PNG và WebP tối đa 5 MB.
- **Dùng camera → Mở camera** cập nhật dự đoán trên khung hình. Chọn **Chụp ảnh → Phân loại ảnh** để xem ba dự đoán cao nhất. Truy cập camera từ điện thoại qua mạng LAN cần HTTPS.

Flask dùng `models/` và đọc số liệu từ `results/`.

## Kiểm tra và huấn luyện lại

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.inspect_dataset
.\.venv\Scripts\python.exe -X utf8 -m src.evaluate
.\.venv\Scripts\python.exe -X utf8 -m src.predict dataset/anhtraicay/data/cam/cam_249.jpg
```

Để huấn luyện một checkpoint riêng từ manifest hiện tại, dùng:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.train_fruit_transfer --manifest results/labels.csv --output-dir results/mobilenet_transfer
.\.venv\Scripts\python.exe -X utf8 -m src.train_cnn --manifest results/labels.csv --output-dir results/cnn_experiment
```

Để huấn luyện bộ lọc trái cây cho một checkpoint riêng, giải nén [COCO128](https://docs.ultralytics.com/datasets/detect/coco128) và dùng lệnh dưới đây. Mô hình dùng ảnh đối chứng từ COCO128 và trọng số ImageNet của MobileNetV2.

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.train_fruit_presence 'đường-dẫn/coco128' --manifest results/labels.csv --output results/mobilenet_transfer/fruit_presence.keras
```

## Chạy notebook

Mở `Nhom_11_Phan_loai_trai_cay_CNN.ipynb` với kernel Python của dự án. Để huấn luyện hoặc kiểm thử lại, chép bộ ảnh vào `dataset/anhtraicay/data/` theo cấu trúc đường dẫn trong `results/labels.csv`. Website có thể chạy thử với các checkpoint trong `models/`.
