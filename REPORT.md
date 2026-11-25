# Báo cáo phân loại trái cây CNN – Nhóm 11

## Dữ liệu hiện tại

Trong `dataset/anhtraicay/data` còn 11 thư mục loại quả, gồm 2.610 ảnh nguồn. Sau khi lọc 53 ảnh trùng nội dung, manifest `results/labels.csv` ghi nhận 2.557 ảnh. Dâu tây, Dưa pepino và Salak đã được loại khỏi bộ nhãn hiện tại vì ảnh của ba loại này thuộc thư mục `test` đã xóa.

Cách chia ảnh của 11 nhãn còn lại được giữ từ lần huấn luyện trước để có thể đối chiếu kết quả. Cột `split` trong manifest xác định ảnh dùng để train, validation hay test. Các ảnh Bưởi rất giống nhau được xếp chung một tập. Chuỗi 162 ảnh Mít của cùng một quả nằm trong tập train. Toàn bộ 7 ảnh test Mít nằm ngoài chuỗi đó.

## Huấn luyện

Ảnh được chuyển thành RGB và đổi kích thước về 224 × 224. Bộ phân loại dùng đặc trưng từ MobileNetV2 đã huấn luyện trước trên ImageNet và học một lớp phân loại Softmax trên dữ liệu hiện tại. Theo kết quả validation, hệ số `C = 0.75` đạt độ chính xác 98,60% trên 358 ảnh. Bộ lọc trái cây dùng ảnh train hiện tại và ảnh đối chứng từ COCO128. Ngưỡng bộ lọc là 0,35.

## Test theo nhãn

Website đoán đúng **342/368 ảnh (92,93%)** ở vị trí đầu. Bộ lọc cho qua cả 368 ảnh trái cây trong tập test và loại đúng 58/58 ảnh đối chứng giữ lại. F1 macro của bộ phân loại là **93,61%**.

| Nhãn | Train | Validation | Test | Website đoán đúng |
|---|---:|---:|---:|---:|
| Bưởi | 91 | 20 | 20 | 20 |
| Cam | 175 | 37 | 38 | 38 |
| Chuối | 175 | 37 | 38 | 38 |
| Dưa hấu | 175 | 37 | 38 | 38 |
| Lê | 175 | 37 | 38 | 33 |
| Mít | 168 | 5 | 7 | 7 |
| Nho | 175 | 37 | 38 | 38 |
| Quýt | 172 | 37 | 37 | 37 |
| Thơm | 175 | 37 | 38 | 38 |
| Táo | 175 | 37 | 38 | 24 |
| Xoài | 175 | 37 | 38 | 31 |
| **Tổng** | **1.831** | **358** | **368** | **342** |

**Táo** có tỷ lệ đúng thấp nhất, 24/38 ảnh. Bộ test được giữ lại từ dữ liệu trong dự án. Một số ảnh có thể là các góc chụp khác nhau của cùng một quả, nên tỷ lệ trên phản ánh bộ test này.

Xem kết quả từng ảnh trong [test_predictions.csv](results/test_predictions.csv), kết quả qua bộ lọc theo nhãn trong [website_test_summary.csv](results/website_test_summary.csv) và các lỗi nhầm lớp trong [confusion_matrix.png](results/confusion_matrix.png). Flask nạp checkpoint mặc định từ `models/` và hiển thị kết quả trong `results/`.
