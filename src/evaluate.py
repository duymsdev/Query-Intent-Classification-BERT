# src/evaluate.py
# Mục đích: đánh giá mô hình đã train trên tập test, in các chỉ số và vẽ Confusion Matrix

import json                                    # lưu chỉ số ra file JSON
import matplotlib                              # thư viện vẽ biểu đồ
matplotlib.use("Agg")                          # chế độ vẽ không cần cửa sổ, chỉ lưu ra file ảnh
import matplotlib.pyplot as plt                # giao diện vẽ
import pandas as pd                            # đọc CSV
import seaborn as sns                          # vẽ heatmap đẹp hơn
import torch                                   # deep learning
from sklearn.metrics import (
    accuracy_score,                            # tính Accuracy
    classification_report,                     # báo cáo Precision/Recall/F1 theo từng nhãn
    confusion_matrix,                          # ma trận nhầm lẫn
    precision_recall_fscore_support,           # Precision/Recall/F1 tổng hợp
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Dùng lại các hằng số và hàm đã viết ở train.py (import không kích hoạt việc train
# nhờ khối if __name__ == "__main__" ở cuối train.py)
from train import DATA_DIR, MAX_LEN, MODEL_DIR, ROOT, segment_text

ASSETS_DIR = ROOT / "assets"      # thư mục lưu hình và báo cáo
BATCH_SIZE = 16                   # số câu dự đoán mỗi lượt


def predict_labels(texts, model, tokenizer, device):
    """Nhận danh sách câu, trả về danh sách ID nhãn mà mô hình dự đoán."""
    model.eval()                                   # chế độ đánh giá (tắt dropout)
    all_preds = []
    with torch.no_grad():                          # không tính gradient -> nhanh hơn, đỡ tốn bộ nhớ
        for i in range(0, len(texts), BATCH_SIZE):
            chunk = [segment_text(t) for t in texts[i:i + BATCH_SIZE]]   # tách từ cho từng câu
            enc = tokenizer(                       # token hóa cả lô, padding theo câu dài nhất trong lô
                chunk, truncation=True, max_length=MAX_LEN,
                padding=True, return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits           # điểm số thô cho 5 nhãn, kích thước [số câu, 5]
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())   # lấy nhãn điểm cao nhất
    return all_preds


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ASSETS_DIR.mkdir(exist_ok=True)                # tạo thư mục assets/ nếu chưa có

    # Đọc tập test và nạp mô hình đã train
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

    # Bảng ánh xạ số -> tên nhãn được lưu sẵn trong config của mô hình
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label_ids = sorted(id2label)                   # [0, 1, 2, 3, 4]
    label_names = [id2label[i] for i in label_ids] # tên nhãn theo đúng thứ tự

    # Dự đoán toàn bộ tập test
    y_true = test_df["label"].tolist()             # nhãn thật
    y_pred = predict_labels(test_df["text"].tolist(), model, tokenizer, device)  # nhãn dự đoán

    # ---------- Tính các chỉ số ----------
    acc = accuracy_score(y_true, y_pred)
    # zero_division=0: nếu một nhãn không có dự đoán nào thì coi chỉ số bằng 0, tránh lỗi chia cho 0
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(                # bảng chi tiết cho từng nhãn
        y_true, y_pred, labels=label_ids, target_names=label_names, digits=4, zero_division=0)

    print(f"Số câu test: {len(y_true)}")
    print(f"Accuracy           : {acc:.4f}")
    print(f"Precision (macro)  : {p_macro:.4f}")
    print(f"Recall    (macro)  : {r_macro:.4f}")
    print(f"F1-score  (macro)  : {f1_macro:.4f}")
    print(f"F1-score  (weighted): {f1_w:.4f}")
    print("\nChi tiết theo từng nhãn:\n" + report)

    # Lưu chỉ số để dùng trong README
    metrics = {
        "num_test_samples": len(y_true),
        "accuracy": round(acc, 4),
        "precision_macro": round(p_macro, 4), "recall_macro": round(r_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(p_w, 4), "recall_weighted": round(r_w, 4),
        "f1_weighted": round(f1_w, 4),
    }
    with open(ASSETS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(ASSETS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # ---------- Vẽ Confusion Matrix ----------
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)   # ma trận 5x5
    plt.figure(figsize=(8, 6))                                # kích thước hình
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",                # annot=True: ghi số vào từng ô
        xticklabels=label_names, yticklabels=label_names,
    )
    plt.xlabel("Predicted label")                             # trục ngang: nhãn dự đoán
    plt.ylabel("True label")                                  # trục dọc: nhãn thật
    plt.title(f"Confusion Matrix (Accuracy = {acc:.2%})")
    plt.xticks(rotation=30, ha="right")                       # xoay chữ cho khỏi chồng nhau
    plt.tight_layout()                                        # tự căn lề cho vừa hình
    plt.savefig(ASSETS_DIR / "confusion_matrix.png", dpi=150) # lưu ảnh
    plt.close()

    # ---------- Liệt kê các câu bị đoán sai ----------
    wrong = [(t, id2label[a], id2label[b])
             for t, a, b in zip(test_df["text"], y_true, y_pred) if a != b]
    print(f"\nSố câu đoán sai: {len(wrong)}")
    for text, true_label, pred_label in wrong:
        print(f"  - '{text}'\n      thật: {true_label} | đoán: {pred_label}")

    print(f"\nĐã lưu kết quả vào: {ASSETS_DIR}")


if __name__ == "__main__":
    main()