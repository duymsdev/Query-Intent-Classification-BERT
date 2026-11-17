# src/preprocess.py
# Mục đích: đọc dữ liệu thô -> làm sạch -> mã hóa nhãn -> chia train/test -> tăng cường dữ liệu train

import json
import random
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------- CẤU HÌNH ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "intent_data.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
HARD_TEST_PATH = DATA_DIR / "hard_test.csv"
LABEL_MAP_PATH = DATA_DIR / "label_map.json"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Bảng viết tắt thường gặp, CHỈ dùng để tăng cường dữ liệu train
# (cố ý không có KHMT, KT để dùng làm "từ chưa từng thấy" khi đánh giá)
ABBREVIATIONS = {
    "công nghệ thông tin": "CNTT",
    "kỹ thuật phần mềm": "KTPM",
    "an toàn thông tin": "ATTT",
    "quản trị kinh doanh": "QTKD",
}


def clean_text(text: str) -> str:
    """Làm sạch một câu: chuẩn hóa Unicode, bỏ ký tự lạ, gộp khoảng trắng."""
    text = unicodedata.normalize("NFC", str(text))       # đưa chữ có dấu về một dạng duy nhất
    text = re.sub(r"[^\w\s?.,%]", " ", text)             # bỏ ký tự đặc biệt (giữ chữ, số, ? . , %)
    text = re.sub(r"\s+", " ", text).strip()             # gộp khoảng trắng thừa
    return text


def remove_diacritics(text: str) -> str:
    """Bỏ dấu tiếng Việt: 'Học phí ngành' -> 'Hoc phi nganh'."""
    text = text.replace("đ", "d").replace("Đ", "D")      # đ/Đ không tự tách dấu được nên thay thủ công
    text = unicodedata.normalize("NFD", text)            # tách chữ cái và dấu thành 2 ký tự riêng
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")  # bỏ các ký tự dấu (Mn = dấu kết hợp)
    return unicodedata.normalize("NFC", text)            # ghép lại về dạng chuẩn


def abbreviate(text: str) -> str:
    """Thay các cụm dài bằng viết tắt: 'Công nghệ thông tin' -> 'CNTT'."""
    for full, short in ABBREVIATIONS.items():
        text = re.sub(full, short, text, flags=re.IGNORECASE)   # không phân biệt hoa/thường
    return text


def augment_train(train_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Tạo thêm biến thể (không dấu, viết tắt) cho MỖI câu trong tập train."""
    rng = random.Random(seed)                            # bộ sinh ngẫu nhiên có hạt giống cố định
    new_rows = []
    for _, row in train_df.iterrows():
        text, intent, label = row["text"], row["intent"], row["label"]

        # Biến thể 1: không dấu, một nửa số câu viết thường hết (giống người gõ vội)
        plain = remove_diacritics(text)
        if rng.random() < 0.5:
            plain = plain.lower()
        new_rows.append({"text": plain, "intent": intent, "label": label})

        # Biến thể 2 và 3: chỉ tạo khi câu có cụm thay được bằng viết tắt
        short = abbreviate(text)
        if short != text:
            new_rows.append({"text": short, "intent": intent, "label": label})
            new_rows.append({"text": remove_diacritics(short).lower(), "intent": intent, "label": label})

    out = pd.concat([train_df, pd.DataFrame(new_rows)], ignore_index=True)   # gộp bản gốc + biến thể
    return out.drop_duplicates(subset=["text"]).reset_index(drop=True)       # bỏ câu trùng


def main():
    df = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
    print(f"[1] Số dòng ban đầu: {len(df)}")

    df = df.dropna(subset=["text", "intent"])
    df["text"] = df["text"].apply(clean_text)
    df["intent"] = df["intent"].str.strip()
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"[2] Số dòng sau khi làm sạch: {len(df)}")
    print("[3] Phân bố nhãn:")
    print(df["intent"].value_counts().to_string())

    labels = sorted(df["intent"].unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    df["label"] = df["intent"].map(label2id)
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
    print(f"[4] Ánh xạ nhãn: {label2id}")

    # Chia train/test TRƯỚC (cùng random_state nên test.csv giống hệt các lần trước)
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_STATE,
    )
    n_train_original = len(train_df)

    # Tăng cường dữ liệu SAU khi chia, và chỉ cho tập train
    train_df = augment_train(train_df, RANDOM_STATE)
    # Chốt chặn rò rỉ: bỏ khỏi train bất kỳ câu nào trùng với câu trong tập test
    train_df = train_df[~train_df["text"].isin(test_df["text"])].reset_index(drop=True)

    train_df.to_csv(TRAIN_PATH, index=False, encoding="utf-8")
    test_df.to_csv(TEST_PATH, index=False, encoding="utf-8")
    print(f"[5] Train gốc: {n_train_original} câu -> sau tăng cường: {len(train_df)} câu | Test: {len(test_df)} câu")

    # Kiểm tra: bộ test khó phải hoàn toàn tách biệt với train
    if HARD_TEST_PATH.exists():
        hard = pd.read_csv(HARD_TEST_PATH, encoding="utf-8-sig")
        overlap = set(hard["text"]) & set(train_df["text"])
        print(f"[6] Số câu của hard_test trùng với train: {len(overlap)} (phải bằng 0)")


if __name__ == "__main__":
    main()