# src/train.py
# Mục đích: fine-tune PhoBERT để phân loại intent, rồi lưu mô hình đã train

import json                                   # đọc file ánh xạ nhãn
import random                                 # sinh số ngẫu nhiên (để cố định seed)
from pathlib import Path                      # xử lý đường dẫn

import numpy as np                            # tính toán mảng số
import pandas as pd                           # đọc file CSV
import torch                                  # framework deep learning
from pyvi import ViTokenizer                  # tách từ tiếng Việt
from torch.utils.data import DataLoader, Dataset   # công cụ nạp dữ liệu theo batch
from tqdm import tqdm                         # thanh tiến trình
from transformers import (
    AutoModelForSequenceClassification,       # PhoBERT + lớp phân loại ở đầu ra
    AutoTokenizer,                            # tokenizer tương ứng với mô hình
    get_linear_schedule_with_warmup,          # bộ điều chỉnh learning rate
)

# ---------- 1. CẤU HÌNH (các "hằng số" của dự án) ----------
MODEL_NAME = "vinai/phobert-base-v2"          # mô hình gốc trên Hugging Face
ROOT = Path(__file__).resolve().parent.parent # thư mục gốc dự án
DATA_DIR = ROOT / "data"                      # thư mục data/
MODEL_DIR = ROOT / "models" / "phobert-intent"  # nơi lưu mô hình sau khi train (đã nằm trong .gitignore)

MAX_LEN = 64         # số token tối đa của 1 câu (câu hỏi ngắn nên 64 là dư)
BATCH_SIZE = 8       # số câu xử lý cùng lúc
EPOCHS = 10          # số lần đi qua toàn bộ tập train
LEARNING_RATE = 3e-5 # bước chỉnh trọng số mỗi lần (nhỏ vì ta chỉ tinh chỉnh, không học lại từ đầu)
SEED = 42            # hạt giống ngẫu nhiên để kết quả lặp lại được


def set_seed(seed: int):
    """Cố định mọi nguồn ngẫu nhiên để mỗi lần chạy cho kết quả giống nhau."""
    random.seed(seed)                         # ngẫu nhiên của Python
    np.random.seed(seed)                      # ngẫu nhiên của NumPy
    torch.manual_seed(seed)                   # ngẫu nhiên của PyTorch (CPU)
    torch.cuda.manual_seed_all(seed)          # ngẫu nhiên của PyTorch (GPU)


def segment_text(text: str) -> str:
    """Tách từ tiếng Việt: 'Công nghệ thông tin' -> 'Công_nghệ_thông_tin'.
    Các file evaluate/predict ở bước sau sẽ dùng lại hàm này."""
    return ViTokenizer.tokenize(text)


# ---------- 2. DATASET: cách nạp 1 mẫu dữ liệu ----------
class IntentDataset(Dataset):
    """Giống một lớp implement interface: PyTorch yêu cầu 2 hàm __len__ và __getitem__."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts            # danh sách các câu
        self.labels = labels          # danh sách nhãn (dạng số)
        self.tokenizer = tokenizer    # tokenizer của PhoBERT
        self.max_len = max_len        # độ dài tối đa

    def __len__(self):
        return len(self.texts)        # tổng số mẫu

    def __getitem__(self, idx):
        text = segment_text(self.texts[idx])          # 1) tách từ
        enc = self.tokenizer(                          # 2) token hóa thành số
            text,
            truncation=True,                           # câu dài quá thì cắt bớt
            max_length=self.max_len,                   # độ dài tối đa
            padding="max_length",                      # câu ngắn thì độn <pad> cho đủ độ dài
            return_tensors="pt",                       # trả về tensor của PyTorch
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),            # dãy ID token (bỏ chiều batch thừa)
            "attention_mask": enc["attention_mask"].squeeze(0),  # mask 1/0 phân biệt chữ thật và <pad>
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),  # nhãn đúng
        }


# ---------- 3. HÀM HUẤN LUYỆN CHÍNH ----------
def main():
    set_seed(SEED)

    # Chọn thiết bị: có GPU thì dùng GPU (nhanh hơn rất nhiều), không thì dùng CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang dùng thiết bị: {device}")

    # Đọc dữ liệu train do preprocess.py tạo ra và bảng ánh xạ nhãn
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    with open(DATA_DIR / "label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    label2id = label_map["label2id"]                                   # chữ -> số
    id2label = {int(k): v for k, v in label_map["id2label"].items()}   # số -> chữ (JSON lưu key là chuỗi nên cần ép int)
    print(f"Số mẫu train: {len(train_df)} | Số nhãn: {len(label2id)}")

    # Tải tokenizer và mô hình PhoBERT từ Hugging Face (lần đầu cần tải vài trăm MB)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),   # số nhãn đầu ra = 5, quyết định kích thước lớp Linear cuối
        id2label=id2label,          # lưu ánh xạ vào config để lúc predict dùng lại
        label2id=label2id,
    )
    model.to(device)                # chuyển mô hình sang GPU/CPU

    # Tạo Dataset và DataLoader (DataLoader tự gom mẫu thành batch và xáo trộn)
    train_ds = IntentDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Optimizer AdamW: thuật toán cập nhật trọng số (phiên bản nâng cao của gradient descent)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS                  # tổng số lần cập nhật trọng số
    scheduler = get_linear_schedule_with_warmup(              # khởi động chậm 10% đầu, sau đó giảm dần learning rate về 0
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # ---------- VÒNG LẶP HUẤN LUYỆN ----------
    for epoch in range(1, EPOCHS + 1):
        model.train()                       # bật chế độ train (bật dropout...)
        total_loss, correct, total = 0.0, 0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            batch = {k: v.to(device) for k, v in batch.items()}   # chuyển dữ liệu sang GPU/CPU

            optimizer.zero_grad()           # xóa gradient của lần trước (nếu không nó sẽ cộng dồn)
            outputs = model(**batch)        # FORWARD: truyền input_ids, attention_mask, labels vào mô hình
            loss = outputs.loss             # độ sai (cross-entropy), mô hình tự tính vì ta đã đưa labels vào
            loss.backward()                 # BACKWARD: tính gradient cho mọi trọng số
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # chặn gradient quá lớn gây bất ổn
            optimizer.step()                # cập nhật trọng số
            scheduler.step()                # cập nhật learning rate

            total_loss += loss.item() * len(batch["labels"])          # cộng dồn loss để tính trung bình
            preds = outputs.logits.argmax(dim=-1)                     # nhãn có điểm cao nhất là dự đoán
            correct += (preds == batch["labels"]).sum().item()        # đếm số câu đoán đúng
            total += len(batch["labels"])

        # In kết quả trung bình của epoch này
        print(f"  -> Epoch {epoch}: loss = {total_loss / total:.4f} | train accuracy = {correct / total:.4f}")

    # ---------- LƯU MÔ HÌNH ----------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)   # tạo thư mục nếu chưa có
    model.save_pretrained(MODEL_DIR)               # lưu trọng số + config
    tokenizer.save_pretrained(MODEL_DIR)           # lưu tokenizer
    print(f"Đã lưu mô hình tại: {MODEL_DIR}")


# Chỉ chạy main() khi chạy trực tiếp file này (import từ file khác thì không tự train)
if __name__ == "__main__":
    main()