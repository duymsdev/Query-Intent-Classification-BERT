# src/evaluate.py
# Mục đích: đánh giá mô hình trên một tập test, in chỉ số và vẽ Confusion Matrix
# Cách dùng:
#   python src/evaluate.py                                   -> chấm tập test.csv như cũ
#   python src/evaluate.py --test-file hard_test.csv --tag hard_before   -> chấm tập khó, lưu kết quả có hậu tố

import argparse                                # đọc tham số dòng lệnh
import json                                    # lưu chỉ số ra JSON
import matplotlib
matplotlib.use("Agg")                          # vẽ ra file ảnh, không cần cửa sổ
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Dùng lại hằng số và hàm từ train.py (import không kích hoạt train nhờ if __name__ == "__main__")
from train import DATA_DIR, MAX_LEN, MODEL_DIR, ROOT, segment_text

ASSETS_DIR = ROOT / "assets"      # nơi lưu kết quả
BATCH_SIZE = 16                   # số câu dự đoán mỗi lượt


def predict_labels(texts, model, tokenizer, device):
    """Trả về (danh sách ID nhãn dự đoán, danh sách độ tin cậy tương ứng)."""
    model.eval()                                   # tắt dropout
    all_preds, all_confs = [], []
    with torch.no_grad():                          # không tính gradient
        for i in range(0, len(texts), BATCH_SIZE):
            chunk = [segment_text(t) for t in texts[i:i + BATCH_SIZE]]   # tách từ
            enc = tokenizer(chunk, truncation=True, max_length=MAX_LEN,
                            padding=True, return_tensors="pt").to(device)
            probs = torch.softmax(model(**enc).logits, dim=-1)           # xác suất 5 nhãn
            conf, pred = probs.max(dim=-1)                               # xác suất cao nhất và nhãn tương ứng
            all_preds.extend(pred.cpu().tolist())
            all_confs.extend(conf.cpu().tolist())
    return all_preds, all_confs


def main():
    # Đọc tham số dòng lệnh
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="test.csv", help="Tên file CSV trong thư mục data/")
    parser.add_argument("--tag", default="", help="Hậu tố thêm vào tên file kết quả")
    args = parser.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""    # ví dụ "_hard_before"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ASSETS_DIR.mkdir(exist_ok=True)

    test_df = pd.read_csv(DATA_DIR / args.test_file)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

    # Ánh xạ nhãn lấy từ config của mô hình (nguồn chuẩn duy nhất)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {k: int(v) for k, v in model.config.label2id.items()}
    label_ids = sorted(id2label)
    label_names = [id2label[i] for i in label_ids]

    # Nhãn thật: ánh xạ từ cột "intent" (dùng được cho cả test.csv lẫn hard_test.csv)
    y_true = test_df["intent"].map(label2id).tolist()
    y_pred, confs = predict_labels(test_df["text"].tolist(), model, tokenizer, device)

    # ---------- Chỉ số ----------
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=label_ids, target_names=label_names, digits=4, zero_division=0)

    print(f"Tập test: {args.test_file} | Số câu: {len(y_true)}")
    print(f"Accuracy            : {acc:.4f}")
    print(f"Precision (macro)   : {p_macro:.4f}")
    print(f"Recall    (macro)   : {r_macro:.4f}")
    print(f"F1-score  (macro)   : {f1_macro:.4f}")
    print(f"F1-score  (weighted): {f1_w:.4f}")
    print("\nChi tiết theo từng nhãn:\n" + report)

    metrics = {
        "test_file": args.test_file, "num_test_samples": len(y_true),
        "accuracy": round(acc, 4),
        "precision_macro": round(p_macro, 4), "recall_macro": round(r_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(p_w, 4), "recall_weighted": round(r_w, 4),
        "f1_weighted": round(f1_w, 4),
    }
    with open(ASSETS_DIR / f"metrics{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(ASSETS_DIR / f"classification_report{suffix}.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # ---------- Confusion Matrix ----------
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix - {args.test_file} (Accuracy = {acc:.2%})")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / f"confusion_matrix{suffix}.png", dpi=150)
    plt.close()

    # ---------- Các câu đoán sai (kèm độ tin cậy) ----------
    wrong = [(t, id2label[a], id2label[b], c)
             for t, a, b, c in zip(test_df["text"], y_true, y_pred, confs) if a != b]
    print(f"\nSố câu đoán sai: {len(wrong)}")
    for text, true_label, pred_label, conf in wrong:
        print(f"  - '{text}'\n      thật: {true_label} | đoán: {pred_label} ({conf:.1%})")

    print(f"\nĐã lưu kết quả vào: {ASSETS_DIR} (hậu tố: '{suffix}')")


if __name__ == "__main__":
    main()