# src/predict.py
# Mục đích: nạp mô hình đã train, nhận câu hỏi từ người dùng và dự đoán intent
# Cách dùng 1 (một câu):  python src/predict.py --text "Học phí ngành CNTT bao nhiêu"
# Cách dùng 2 (nhập liên tục): python src/predict.py

import argparse                                 # đọc tham số dòng lệnh (--text ...)
import sys                                      # thoát chương trình khi có lỗi

import torch                                    # deep learning
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Dùng lại hằng số và hàm tách từ ở train.py để đầu vào được xử lý giống hệt lúc huấn luyện
from train import MAX_LEN, MODEL_DIR, segment_text

CONFIDENCE_THRESHOLD = 0.5   # dưới ngưỡng này thì coi là "không chắc chắn"


def load_model():
    """Nạp mô hình, tokenizer và bảng ánh xạ nhãn từ thư mục đã lưu ở bước train."""
    if not MODEL_DIR.exists():                  # chưa train thì chưa có mô hình để nạp
        print(f"Không tìm thấy mô hình tại {MODEL_DIR}. Hãy chạy 'python src/train.py' trước.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()                                # chế độ dự đoán (tắt dropout)
    # Ánh xạ số -> tên nhãn nằm sẵn trong config của mô hình
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    return model, tokenizer, id2label, device


def predict(text, model, tokenizer, id2label, device):
    """Dự đoán intent cho 1 câu. Trả về (nhãn tốt nhất, độ tin cậy, bảng xếp hạng cả 5 nhãn)."""
    enc = tokenizer(
        segment_text(text),                     # tách từ trước, giống lúc train
        truncation=True, max_length=MAX_LEN, return_tensors="pt",
    ).to(device)

    with torch.no_grad():                       # không cần gradient khi dự đoán
        logits = model(**enc).logits            # 5 điểm số thô, kích thước [1, 5]

    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()   # chuyển thành 5 xác suất (tổng = 1)
    best_id = int(probs.argmax())               # vị trí của xác suất lớn nhất

    # Bảng xếp hạng: danh sách (tên nhãn, xác suất) sắp xếp giảm dần
    ranking = sorted(
        ((id2label[i], float(p)) for i, p in enumerate(probs)),
        key=lambda x: x[1], reverse=True,
    )
    return id2label[best_id], float(probs[best_id]), ranking


def show_result(text, label, confidence, ranking):
    """In kết quả ra màn hình cho dễ đọc."""
    print(f'\nCâu hỏi: "{text}"')
    if confidence < CONFIDENCE_THRESHOLD:       # độ tin cậy thấp -> cảnh báo
        print(f"=> Không chắc chắn (gần nhất: {label}, {confidence:.1%}). "
              "Có thể câu hỏi nằm ngoài phạm vi tuyển sinh.")
    else:
        print(f"=> Ý định dự đoán: {label} (độ tin cậy {confidence:.1%})")
    print("   Chi tiết xác suất:")
    for name, p in ranking:
        bar = "#" * int(p * 30)                 # thanh ngang minh họa, dài tối đa 30 ký tự
        print(f"   {name:<22} {p:6.1%} {bar}")


def main():
    parser = argparse.ArgumentParser(description="Dự đoán intent của câu hỏi tuyển sinh")
    parser.add_argument("--text", type=str, default=None, help="Câu hỏi cần dự đoán (bỏ trống để nhập liên tục)")
    args = parser.parse_args()

    model, tokenizer, id2label, device = load_model()

    # Chế độ 1: dự đoán đúng một câu rồi thoát
    if args.text:
        label, conf, ranking = predict(args.text, model, tokenizer, id2label, device)
        show_result(args.text, label, conf, ranking)
        return

    # Chế độ 2: vòng lặp nhập liên tục cho đến khi gõ q
    print("Nhập câu hỏi để dự đoán ý định. Gõ 'q' để thoát.")
    while True:
        try:
            text = input("\nBạn hỏi: ").strip()   # đọc một dòng người dùng gõ
        except EOFError:                        # môi trường không cho nhập (ví dụ chạy bằng !python trong Colab)
            print("Không nhận được dữ liệu nhập. Hãy dùng --text hoặc %run trong Colab.")
            break
        if text.lower() in {"q", "quit", "exit"}:
            print("Tạm biệt!")
            break
        if not text:                            # bỏ qua dòng trống
            continue
        label, conf, ranking = predict(text, model, tokenizer, id2label, device)
        show_result(text, label, conf, ranking)


if __name__ == "__main__":
    main()