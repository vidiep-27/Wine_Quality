# Phân loại chất lượng rượu vang bằng học máy tổ hợp

Mã nguồn thực nghiệm cho học phần Học Máy.
Nội dung phương pháp và phân tích kết quả được trình bày trong `report/`.

## Cài đặt

Sử dụng Python 3.13.5. Cài thư viện và mở JupyterLab:

```bash
python -m pip install -r requirements.txt
jupyter lab
```

## Thứ tự chạy

| Bước | Notebook | Người phụ trách |
|---|---|---|
| 1 | `eda.ipynb` | Vi |
| 2 | `chuanbi_dulieu.ipynb` | Quân |
| 3 | `baseline_dt_bagging.ipynb` | Vi |
| 4 | `rforest_extratress.ipynb` | Quân |
| 5 | `final_evaluation.ipynb` | Vi tổng hợp, Quân phối hợp |

Hai thành viên dùng cùng dữ liệu chia tập, cấu hình và phiên bản thư viện.

Các module trong `src/` được notebook gọi trực tiếp, không cần chạy riêng.

## Kết quả đầu ra

Kết quả được lưu trong `output/`:

- `test_metrics.json`: các chỉ số đánh giá test.
- `test_predictions.csv`: dự đoán từng mẫu test.
- `test_classification_report.csv`: chỉ số theo lớp.
- `final_pipeline.joblib`: Pipeline cuối đã huấn luyện.
- `figures/`: biểu đồ kết quả.

## Lưu ý

Notebook tổng hợp có ô đọc lại kết quả đã lưu.
Không cần huấn luyện lại khi chỉ xem bảng và biểu đồ.
Không điều chỉnh mô hình dựa trên kết quả test.