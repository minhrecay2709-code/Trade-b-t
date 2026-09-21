name: Trade Bot

on:
  schedule:
    - cron: '0 2-4,6-7 * * 1-5'
  workflow_dispatch:

concurrency:
  group: trade-bot
  cancel-in-progress: false

permissions:
  contents: write

jobs:
  quet-tin-hieu:
    runs-on: ubuntu-latest
    timeout-minutes: 90
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python bot.py
        env:
          EMAIL_USER: ${{ secrets.EMAIL_USER }}
          EMAIL_PASS: ${{ secrets.EMAIL_PASS }}
          EMAIL_TO: ${{ secrets.EMAIL_TO }}
      - name: Lưu trạng thái đã gửi vào repo
        run: |
          git config user.name "trade-bot"
          git config user.email "trade-bot@users.noreply.github.com"
          git add trang_thai_da_gui.json
          git diff --staged --quiet && echo "Không có gì thay đổi" || (git commit -m "Cập nhật trạng thái đã gửi" && git push)
"""
BOT QUÉT TÍN HIỆU + GỬI EMAIL
Bản mở rộng: quét toàn bộ sàn HOSE (không chỉ VN100), chấm điểm 4 yếu tố
của Churning (gửi từ 2/4 trở lên, ghi rõ yếu tố nào thiếu), bỏ TIP/MCP,
thêm HHP vào danh sách theo dõi thủ công.
"""

from datetime import date, timedelta
import os
import time
import json
import smtplib
from email.mime.text import MIMEText

import numpy as np
import pandas as pd

from vnstock import Quote, Listing

# ============ CẤU HÌNH ============
DS_BO_SUNG_THU_CONG = ["HHP"]  # mã muốn theo dõi thêm dù chưa chắc vào nhóm quét tự động

THOI_GIAN_NGHI_GIUA_MA = 6
THOI_GIAN_NGHI_KHI_BI_CHAN = 20
SO_LAN_THU_LAI = 2
FILE_TRANG_THAI = "trang_thai_da_gui.json"

HE_SO_ATR_CAT_LO = 1.5
TY_LE_RR = 2.0
KHOI_LUONG_TB_TOI_THIEU = 100_000
SO_YEU_TO_TOI_THIEU_DE_GUI = 2  # gửi email khi đạt từ 2/4 yếu tố trở lên

NGAY_KET_THUC = date.today()
NGAY_BAT_DAU = NGAY_KET_THUC - timedelta(days=100)


# ============ LẤY DANH SÁCH MÃ (toàn bộ HOSE, có dự phòng nếu lỗi) ============
def lay_danh_sach_ma():
    try:
        listing = Listing()
        try:
            ds = listing.symbols_by_exchange("HOSE")
            ds = list(ds["symbol"]) if hasattr(ds, "columns") else list(ds)
        except Exception:
            ds = listing.symbols_by_group("VN100")
        ds_day_du = list(dict.fromkeys(list(ds) + DS_BO_SUNG_THU_CONG))
        print(f"Lấy được {len(ds_day_du)} mã để quét (đã gộp thêm {DS_BO_SUNG_THU_CONG}).")
        return ds_day_du
    except Exception as e:
        print(f"Không lấy được danh sách từ vnstock ({e}), dùng danh sách dự phòng.")
        return DS_BO_SUNG_THU_CONG + [
            "HPG", "MWG", "VCB", "VHM", "VIC", "GAS", "MSN", "TCB",
            "CTG", "BID", "VPB", "MBB", "ACB", "STB", "SSI", "VRE", "PLX", "POW",
            "GVR", "SAB", "HDB", "TPB", "BVH", "KDH", "PDR", "NVL", "DGC", "VJC",
        ]


# ============ ĐỌC / GHI TRẠNG THÁI ĐÃ GỬI (chống trùng lặp) ============
def doc_trang_thai_da_gui() -> dict:
    if os.path.exists(FILE_TRANG_THAI):
        try:
            with open(FILE_TRANG_THAI, "r", encoding="utf-8") as f:
                du_lieu = json.load(f)
        except Exception:
            du_lieu = {}
    else:
        du_lieu = {}
    hom_nay = str(date.today())
    return du_lieu.get(hom_nay, {})


def ghi_trang_thai_da_gui(da_gui_hom_nay: dict):
    hom_nay = str(date.today())
    with open(FILE_TRANG_THAI, "w", encoding="utf-8") as f:
        json.dump({hom_nay: da_gui_hom_nay}, f, ensure_ascii=False, indent=2)


# ============ LẤY DỮ LIỆU MỘT MÃ ============
def lay_du_lieu(ma: str) -> pd.DataFrame:
    q = Quote(symbol=ma, source="KBS")
    df = q.history(
        start=NGAY_BAT_DAU.strftime("%Y-%m-%d"),
        end=NGAY_KET_THUC.strftime("%Y-%m-%d"),
        interval="1D",
    )
    return df.reset_index(drop=True)


# ============ KIỂM TRA BỐI CẢNH THỊ TRƯỜNG (VN-INDEX) ============
def kiem_tra_thi_truong() -> tuple:
    try:
        df = lay_du_lieu("VNINDEX")
        if df is None or len(df) < 25:
            return True, "Không đủ dữ liệu VN-Index để đánh giá, bỏ qua bộ lọc."
        df = df.copy()
        df["ma20"] = df["close"].rolling(20).mean()
        row = df.iloc[-1]
        if pd.isna(row["ma20"]):
            return True, "MA20 của VN-Index chưa tính được, bỏ qua bộ lọc."
        dang_tren_ma20 = row["close"] >= row["ma20"]
        pct_so_ma = (row["close"] - row["ma20"]) / row["ma20"] * 100
        ghi_chu = (
            f"VN-Index: {row['close']:.2f} | MA20: {row['ma20']:.2f} | "
            f"Cách MA20: {pct_so_ma:+.2f}%"
        )
        return bool(dang_tren_ma20), ghi_chu
    except Exception as e:
        print(f"Lỗi kiểm tra VN-Index: {e}")
        return True, "Lỗi khi lấy dữ liệu VN-Index, bỏ qua bộ lọc (mặc định cho phép)."


def tinh_rsi(close: pd.Series, chu_ky: int = 14) -> pd.Series:
    thay_doi = close.diff()
    tang = thay_doi.clip(lower=0)
    giam = -thay_doi.clip(upper=0)
    tb_tang = tang.ewm(alpha=1 / chu_ky, adjust=False, min_periods=chu_ky).mean()
    tb_giam = giam.ewm(alpha=1 / chu_ky, adjust=False, min_periods=chu_ky).mean()
    rs = tb_tang / tb_giam.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def tinh_atr(df: pd.DataFrame, chu_ky: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close_truoc = (df["high"] - df["close"].shift(1)).abs()
    low_close_truoc = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close_truoc, low_close_truoc], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / chu_ky, adjust=False, min_periods=chu_ky).mean()


def tinh_cong_thuc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bien_do"] = df["high"] - df["low"]
    df["than_pct"] = (df["close"] - df["open"]).abs() / df["bien_do"].replace(0, np.nan) * 100
    df["bong_tren_pct"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["bien_do"].replace(0, np.nan) * 100
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]
    df["dinh_cuc_bo"] = df["high"].shift(1).rolling(20).max()
    df["day_cuc_bo"] = df["low"].shift(1).rolling(20).min()
    df["rsi14"] = tinh_rsi(df["close"], 14)
    df["atr14"] = tinh_atr(df, 14)
    return df


# ============ CHẤM ĐIỂM 4 YẾU TỐ CỦA CHURNING ============
def danh_gia_churning(row) -> dict:
    thanh_khoan_du = pd.notna(row.get("vol_ma20")) and row["vol_ma20"] >= KHOI_LUONG_TB_TOI_THIEU
    gan_dinh = (
        pd.notna(row.get("dinh_cuc_bo"))
        and row["dinh_cuc_bo"] != 0
        and abs(row["close"] - row["dinh_cuc_bo"]) / row["dinh_cuc_bo"] <= 0.03
    )
    boi_canh_hop_le = thanh_khoan_du and gan_dinh

    yeu_to = {
        "Thân nến nhỏ": pd.notna(row.get("than_pct")) and row["than_pct"] < 30,
        "Bóng trên dài": pd.notna(row.get("bong_tren_pct")) and row["bong_tren_pct"] > 40,
        "Volume đột biến": pd.notna(row.get("vol_ratio")) and row["vol_ratio"] > 1.5,
        "RSI quá mua": pd.notna(row.get("rsi14")) and row["rsi14"] > 70,
    }
    so_dat = sum(1 for v in yeu_to.values() if v)
    return {
        "boi_canh_hop_le": boi_canh_hop_le,
        "yeu_to": yeu_to,
        "so_dat": so_dat,
    }


def tinh_diem_cat_lo_chot_loi(row) -> tuple:
    entry = row["close"]
    if pd.notna(row.get("atr14")):
        stop_loss_atr = entry - HE_SO_ATR_CAT_LO * row["atr14"]
    else:
        stop_loss_atr = None
    if pd.notna(row.get("day_cuc_bo")):
        stop_loss_day = min(row["day_cuc_bo"], row["low"])
    else:
        stop_loss_day = row["low"] * 0.97
    stop_loss = max(stop_loss_atr, stop_loss_day) if stop_loss_atr is not None else stop_loss_day
    risk = entry - stop_loss
    take_profit = entry + risk * TY_LE_RR
    return round(entry, 2), round(stop_loss, 2), round(take_profit, 2)


def gui_email(noi_dung: str):
    email_user = os.environ.get("EMAIL_USER", "").strip()
    email_pass = os.environ.get("EMAIL_PASS", "").strip()
    email_to = os.environ.get("EMAIL_TO", "").strip()
    if not email_user or not email_pass or not email_to:
        raise RuntimeError("Thiếu secret: kiểm tra EMAIL_USER, EMAIL_PASS, EMAIL_TO.")

    msg = MIMEText(noi_dung, "plain", "utf-8")
    msg["Subject"] = f"[Trade Bot] Tín hiệu {date.today()}"
    msg["From"] = email_user
    msg["To"] = email_to

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(email_user, email_pass)
            server.sendmail(email_user, [email_to], msg.as_string())
    except smtplib.SMTPAuthenticationError:
        raise RuntimeError("Gmail từ chối đăng nhập. Kiểm tra lại EMAIL_USER và EMAIL_PASS.")


def main():
    danh_sach_ma = lay_danh_sach_ma()
    da_gui_hom_nay = doc_trang_thai_da_gui()

    time.sleep(THOI_GIAN_NGHI_GIUA_MA)
    thi_truong_thuan_loi, ghi_chu_thi_truong = kiem_tra_thi_truong()
    print(f"Bối cảnh thị trường: {ghi_chu_thi_truong}")
    if not thi_truong_thuan_loi:
        print("VN-Index đang dưới MA20 — tạm ngưng gửi tín hiệu mua mới.")
        return

    ket_qua_email = []
    co_tin_hieu_moi = False

    for idx, ma in enumerate(danh_sach_ma):
        thanh_cong = False
        so_lan_thu = 0
        while not thanh_cong and so_lan_thu < SO_LAN_THU_LAI:
            try:
                time.sleep(THOI_GIAN_NGHI_GIUA_MA)
                df = lay_du_lieu(ma)
                if df is None or len(df) < 25:
                    thanh_cong = True
                    continue
                df = tinh_cong_thuc(df)
                row = df.iloc[-1]
                danh_gia = danh_gia_churning(row)

                if danh_gia["boi_canh_hop_le"] and danh_gia["so_dat"] >= SO_YEU_TO_TOI_THIEU_DE_GUI:
                    khoa = f"{ma}_churning"
                    if khoa in da_gui_hom_nay:
                        print(f"Bỏ qua {ma} (đã gửi hôm nay rồi)")
                    else:
                        entry, sl, tp = tinh_diem_cat_lo_chot_loi(row)
                        dat = [k for k, v in danh_gia["yeu_to"].items() if v]
                        thieu = [k for k, v in danh_gia["yeu_to"].items() if not v]
                        rsi_str = f"{row['rsi14']:.1f}" if pd.notna(row.get("rsi14")) else "N/A"

                        khoi = (
                            f"{ma} — Churning ({danh_gia['so_dat']}/4 yếu tố)\n"
                            f"  Giá hiện tại: {entry}\n"
                            f"  RSI14: {rsi_str}\n"
                            f"  Cắt lỗ: {sl}\n"
                            f"  Chốt lời: {tp}\n"
                            f"  Đạt: {', '.join(dat) if dat else '(không có)'}\n"
                            f"  Thiếu: {', '.join(thieu) if thieu else '(không thiếu gì — đủ 4/4)'}\n"
                        )
                        ket_qua_email.append(khoi)
                        da_gui_hom_nay[khoa] = True
                        co_tin_hieu_moi = True
                thanh_cong = True
            except Exception as e:
                so_lan_thu += 1
                print(f"Lỗi mã {ma} (lần {so_lan_thu}): {e}")
                if "limit" in str(e).lower() or "rate" in str(e).lower():
                    print(f"Bị giới hạn API, đợi {THOI_GIAN_NGHI_KHI_BI_CHAN} giây rồi thử lại...")
                    time.sleep(THOI_GIAN_NGHI_KHI_BI_CHAN)
                else:
                    break

        if (idx + 1) % 50 == 0:
            print(f"Đã quét {idx + 1}/{len(danh_sach_ma)} mã...")

    if ket_qua_email:
        noi_dung = f"{ghi_chu_thi_truong}\n\n" + "\n".join(ket_qua_email)
        noi_dung += (
            "\n\n(Lưu ý: đây là tín hiệu tự động, không phải khuyến nghị đầu tư. "
            "Mã đạt càng nhiều yếu tố (gần 4/4) thì mức độ tin cậy theo backtest càng cao. "
            "Luôn tự kiểm tra lại biểu đồ, đặc biệt các yếu tố còn thiếu, trước khi quyết định.)"
        )
        try:
            gui_email(noi_dung)
            print("Đã gửi email với", len(ket_qua_email), "tín hiệu mới.")
            if co_tin_hieu_moi:
                ghi_trang_thai_da_gui(da_gui_hom_nay)
        except Exception as e:
            print(f"Gửi email thất bại: {e}")
            print("Sẽ tự động thử gửi lại ở lần chạy kế tiếp.")
    else:
        print("Không có tín hiệu mới lúc này.")


if __name__ == "__main__":
    main()
