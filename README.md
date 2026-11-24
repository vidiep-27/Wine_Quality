# Phân loại chất lượng rượu vang bằng học máy tổ hợp

Mã nguồn thực nghiệm cho học phần Học Máy.

## Cài đặt

Môi trường chính sử dụng Python 3.13.5. Metadata của notebook cải tiến ghi Python 3.12.3. Cài thư viện và mở JupyterLab bằng các lệnh sau.

```bash
python -m pip install -r requirements.txt
jupyter lab
```

## Thứ tự chạy

| Bước | Notebook | Nội dung |
|---|---|---|
| 1 | `eda.ipynb` | Khảo sát, làm sạch dữ liệu và tạo nhãn ba mức |
| 2 | `chuanbi_dulieu.ipynb` | Chọn 12 đặc trưng, mã hóa loại rượu, chia train/test |
| 3 | `baseline_dt_bagging.ipynb` | Mô hình cơ sở, Decision Tree, Bagging: CV và tìm siêu tham số |
| 4 | `rforest_extratress.ipynb` | Random Forest, Extra Trees: CV và tìm siêu tham số |
| 5 | `final_evaluation.ipynb` | Chọn mô hình theo CV, đánh giá test, quá khớp, độ quan trọng đặc trưng, OOB |
| 6 | `caitien_ablation.ipynb` | Thử A1–A4 trên train, chọn theo CV và đọc kết quả test đã lưu |

Hai thành viên dùng cùng dữ liệu chia tập, cấu hình và phiên bản thư viện.

`src/improvement.py` chứa các hàm dùng cho bước 6 và hàm tạo kết quả test cho ứng viên cải tiến.

## Notebook cải tiến (`caitien_ablation.ipynb`)

Chạy sau `final_evaluation.ipynb`. Chỉ cải tiến mô hình đã chốt (Extra Trees ban đầu), không huấn luyện lại thuật toán khác.

### Các cấu hình cải tiến

| Mã | Thay đổi so với Extra Trees ban đầu |
|---|---|
| A1 | Thêm đặc trưng `free_so2_ratio` (SO2 tự do / SO2 tổng), rồi chọn đặc trưng bằng `SelectFromModel` (giữ biến có độ quan trọng từ trung vị trở lên) trong Pipeline |
| A2 | `class_weight='balanced'` và `min_samples_leaf=3` |
| A3 | Điều chỉnh quyết định cho lớp Cao: nhân xác suất dự đoán của lớp Cao với hệ số `w`, giữ nguyên xác suất hai lớp còn lại, rồi chọn lớp có giá trị lớn nhất. `w` thuộc {1,0; 1,25; 1,5; 1,75; 2,0; 2,5}, chọn bằng CV lồng 3 fold bên trong dữ liệu huấn luyện, tối đa hóa F1 macro. Mô hình cây không đổi, chỉ đổi cách chọn nhãn nên lớp Cao dễ được dự đoán hơn |
| A4 | Kết hợp A2 và A3 (phép thử xem hai cách có bổ trợ nhau không) |

### Quy trình

1. Chẩn đoán A0 từ tập huấn luyện bằng chênh lệch F1 macro giữa train và validation, cùng lỗi dự đoán ngoài fold của lớp Cao. A0 nhận đúng 343/807 mẫu Cao, tương ứng recall tính gộp 0,4250. Recall trung bình qua năm fold là 0,4251.
2. Chạy A0–A4 trên train với đúng năm fold từ `output/cv_folds.joblib`. Mốc A0 phải khớp điểm CV trong `selection_decision.json`. Chỉ số chính là F1 macro. Theo dõi thêm recall và precision của lớp Cao.
3. Chọn ứng viên có mức tăng F1 macro trung bình dương và tốt hơn A0 ở ít nhất 4/5 fold. Nếu có nhiều cấu hình đạt tiêu chí, chọn cấu hình có F1 macro CV cao nhất. Kết quả đã lưu chọn A3.
4. Đọc các bảng test đã lưu để đối chiếu A0 và A3 trên cùng tập kiểm tra.

Mặc định `RUN = False` để xem kết quả đã lưu. Đặt `RUN = True` để tính lại ablation và đường cong kiểm định trên train. Sau khi có kết quả CV, nếu dự án chưa có kết quả test cho ứng viên cải tiến, chạy lệnh sau từ thư mục gốc dự án.

```bash
python -c "from src.improvement import evaluate_improved; evaluate_improved('.')"
```

Hàm chọn ứng viên theo CV, khớp trên toàn bộ train và ghi `improved_decision.json` trước khi đọc test. Khi tệp quyết định đã tồn tại, dùng lại kết quả test đã lưu. Nếu các tệp đầu ra chưa đủ, kiểm tra lần đánh giá đã hoàn tất hay chưa.

## Kết quả đầu ra

Kết quả được lưu trong `output/`:

- `test_metrics.json`: các chỉ số đánh giá test của Extra Trees ban đầu.
- `test_predictions.csv`: dự đoán từng mẫu test.
- `test_classification_report.csv`: chỉ số theo lớp.
- `test_confusion_matrix.csv`: ma trận nhầm lẫn trên test của Extra Trees ban đầu.
- `final_pipeline.joblib`: Pipeline cuối đã huấn luyện.
- `ablation_results.csv`, `ablation_fold_results.csv`: kết quả A1–A4 trung bình và theo từng fold.
- `ablation_selected_features.csv`: đặc trưng được giữ ở từng fold (A1).
- `ablation_decision_weights.csv`: hệ số `w` được chọn ở từng fold (A3, A4).
- `validation_curve_extra_trees.csv`: đường cong kiểm định theo `min_samples_leaf` và `max_depth`.
- `improved_decision.json`: quyết định chọn A3, ghi trước khi nạp test.
- `improved_test_metrics.json`, `improved_test_classification_report.csv`, `improved_test_confusion_matrix.csv`: kết quả đối chiếu của A3 trên test.
- `improved_pipeline.joblib`: Pipeline A3 được khớp trên toàn bộ train.
- `figures/`: biểu đồ kết quả.

## Lưu ý

Extra Trees ban đầu (`final_pipeline.joblib`) là mô hình chính đã chốt bằng CV ở vòng so sánh ban đầu. A3 là ứng viên được chọn bằng CV ở vòng cải tiến. Kết quả test dùng để mô tả sự đánh đổi giữa các lớp sau khi đã chốt hai cấu hình.
