# src/preprocess.py
# Mục đích: đọc dữ liệu thô -> làm sạch -> mã hóa nhãn -> chia train/test

import json                      # lưu bảng ánh xạ nhãn <-> số ra file JSON
import re                        # regex: tìm và thay thế ký tự theo mẫu
import unicodedata               # chuẩn hóa Unicode của chữ tiếng Việt
from pathlib import Path         # xử lý đường dẫn file, chạy được trên cả Windows lẫn Linux

import pandas as pd              # đọc và xử lý bảng dữ liệu (như một mảng 2 chiều có tên cột)
from sklearn.model_selection import train_test_split  # hàm chia train/test

# ---------- 1. CẤU HÌNH ĐƯỜNG DẪN ----------
# __file__ là đường dẫn file này (src/preprocess.py); .parent.parent lùi 2 cấp để về thư mục gốc dự án
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"                       # thư mục data/
RAW_PATH = DATA_DIR / "intent_data.csv"        # file dữ liệu thô
TRAIN_PATH = DATA_DIR / "train.csv"            # file train sẽ tạo ra
TEST_PATH = DATA_DIR / "test.csv"              # file test sẽ tạo ra
LABEL_MAP_PATH = DATA_DIR / "label_map.json"   # file lưu ánh xạ nhãn

TEST_SIZE = 0.2        # 20% dữ liệu dùng làm tập test
RANDOM_STATE = 42      # "hạt giống" ngẫu nhiên: cố định để mỗi lần chạy cho cùng kết quả chia


# ---------- 2. HÀM LÀM SẠCH MỘT CÂU ----------
def clean_text(text: str) -> str:
    """Làm sạch một câu văn bản và trả về câu đã sạch."""
    # Ép về kiểu chuỗi (phòng trường hợp ô dữ liệu là số hoặc rỗng)
    text = str(text)
    # Chuẩn hóa Unicode về dạng NFC: cùng một chữ "ế" có thể được lưu bằng 1 hoặc 2 ký tự,
    # NFC đưa tất cả về 1 dạng duy nhất để máy không nhầm chúng là 2 chữ khác nhau
    text = unicodedata.normalize("NFC", text)
    # Thay mọi ký tự KHÔNG phải chữ/số/khoảng trắng/dấu ? . , % bằng khoảng trắng
    # \w khớp chữ cái (kể cả tiếng Việt có dấu) và chữ số; \s khớp khoảng trắng
    text = re.sub(r"[^\w\s?.,%]", " ", text)
    # Gộp nhiều khoảng trắng liên tiếp thành 1, rồi cắt khoảng trắng đầu và cuối câu
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------- 3. HÀM CHÍNH ----------
def main():
    # Đọc CSV. "utf-8-sig" tự bỏ ký tự BOM nếu file được lưu từ Excel/Notepad
    df = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
    print(f"[1] Số dòng ban đầu: {len(df)}")

    # Bỏ các dòng thiếu text hoặc thiếu nhãn
    df = df.dropna(subset=["text", "intent"])

    # Áp dụng hàm clean_text cho từng câu trong cột text
    df["text"] = df["text"].apply(clean_text)
    # Nhãn cũng được cắt khoảng trắng thừa cho chắc
    df["intent"] = df["intent"].str.strip()

    # Bỏ các câu rỗng sau khi làm sạch
    df = df[df["text"] != ""]
    # Bỏ câu trùng lặp (trùng nội dung thì giữ lại một)
    # Lý do: nếu cùng một câu nằm ở cả train và test thì điểm đánh giá sẽ bị "ảo"
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"[2] Số dòng sau khi làm sạch: {len(df)}")

    # Thống kê số câu của từng nhãn để kiểm tra dữ liệu có cân bằng không
    print("[3] Phân bố nhãn:")
    print(df["intent"].value_counts().to_string())

    # ---------- Mã hóa nhãn: chữ -> số ----------
    labels = sorted(df["intent"].unique())                        # danh sách nhãn, sắp xếp cho cố định
    label2id = {label: idx for idx, label in enumerate(labels)}   # ví dụ {"diem_chuan": 0, ...}
    id2label = {idx: label for label, idx in label2id.items()}    # ánh xạ ngược: số -> chữ
    df["label"] = df["intent"].map(label2id)                      # tạo cột label dạng số

    # Lưu ánh xạ ra file để train.py và predict.py dùng lại
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
    print(f"[4] Ánh xạ nhãn: {label2id}")

    # ---------- Chia train / test ----------
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,        # tỉ lệ test = 20%
        stratify=df["label"],       # chia phân tầng: mỗi nhãn đều chia theo đúng tỉ lệ 80/20
        random_state=RANDOM_STATE,  # cố định hạt giống ngẫu nhiên
    )

    # Lưu 2 tập ra file CSV
    train_df.to_csv(TRAIN_PATH, index=False, encoding="utf-8")
    test_df.to_csv(TEST_PATH, index=False, encoding="utf-8")
    print(f"[5] Tập train: {len(train_df)} câu | Tập test: {len(test_df)} câu")
    print(f"    Đã lưu: {TRAIN_PATH.name}, {TEST_PATH.name}, {LABEL_MAP_PATH.name}")


# Chỉ chạy main() khi file được chạy trực tiếp (giống hàm main() trong C++/Java)
if __name__ == "__main__":
    main()