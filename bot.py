"""
BOT QUÉT TÍN HIỆU + GỬI EMAIL — chạy tự động qua GitHub Actions
"""

from datetime import date, timedelta
import os
import smtplib
from email.mime.text import MIMEText

import numpy as np
import pandas as pd

from vnstock import Quote

# ============ CẤU HÌNH ============
MA_CO_PHIEU = [
    "FPT", "VNM", "HPG", "MWG", "VCB", "VHM", "VIC", "GAS", "MSN", "TCB",
    "CTG", "BID", "VPB", "MBB", "ACB", "STB", "SSI", "VRE", "PLX", "POW",
    "GVR", "SAB", "HDB", "TPB", "BVH", "KDH", "PDR", "NVL", "DGC", "VJC",
]

RUI_RO_R = 1.0  # tỷ lệ R:R mặc định 1:2 (stoploss 1R, chốt lời 2R)

NGAY_KET_THUC = date.today()
NGAY_BAT_DAU = NGAY_KET_THUC - timedelta(days=100)


# ============ LẤY DỮ LIỆU ============
def lay_du_lieu(ma: str) -> pd.DataFrame:
    q = Quote(symbol=ma, source="KBS")
    df = q.history(
        start=NGAY_BAT_DAU.strftime("%Y-%m-%d"),
        end=NGAY_KET_THUC.strftime("%Y-%m-%d"),
        interval="1D",
    )
    return df.reset_index(drop=True)


# ============ TÍNH CÔNG THỨC ============
def tinh_cong_thuc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bien_do"] = df["high"] - df["low"]
    df["than_pct"] = (df["close"] - df["open"]).abs() / df["bien_do"].replace(0, np.nan) * 100
    df["bong_tren_pct"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["bien_do"].replace(0, np.nan) * 100
    df["bong_duoi_pct"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["bien_do"].replace(0, np.nan) * 100

    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]

    df["ma20"] = df["close"].rolling(20).mean()
    df["do_doc_ma_pct"] = (df["ma20"] - df["ma20"].shift(5)) / df["ma20"].shift(5) * 100

    df["dinh_cuc_bo"] = df["high"].shift(1).rolling(20).max()
    df["day_cuc_bo"] = df["low"].shift(1).rolling(20).min()

    return df


# ============ PHÁT HIỆN TÍN HIỆU ============
def phat_hien_tin_hieu(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    gia_tren_ma = df["close"] >= df["ma20"]
    gia_gan_ma = (df["close"] - df["ma20"]).abs() / df["ma20"] <= 0.03
    df["tin_hieu_nen_nen"] = (
        (df["than_pct"] < 30) & (df["vol_ratio"] < 0.5) & gia_tren_ma
        & gia_gan_ma & (df["do_doc_ma_pct"] > 0)
    )

    than_nho = df["than_pct"] < 40
    bong_duoi_dai = df["bong_duoi_pct"] > 50
    gan_ma = (df["close"] - df["ma20"]).abs() / df["ma20"] <= 0.02
    df["tin_hieu_test_ma"] = (
        (than_nho | bong_duoi_dai) & (df["vol_ratio"].between(0.5, 0.7)) & gan_ma
    )

    than_lon = df["than_pct"] > 70
    dong_cua_cao = (df["high"] - df["close"]) / df["bien_do"].replace(0, np.nan) <= 0.10
    vuot_dinh = df["close"] > df["dinh_cuc_bo"]
    vuot_ma = df["close"] > df["ma20"]
    vol_no = df["vol_ratio"] > 2.0
    df["tin_hieu_breakout"] = than_lon & dong_cua_cao & vuot_dinh & vuot_ma & vol_no

    gan_dinh = (df["close"] - df["dinh_cuc_bo"]).abs() / df["dinh_cuc_bo"] <= 0.03
    df["tin_hieu_churning"] = (
        (df["than_pct"] < 30) & (df["bong_tren_pct"] > 40) & (df["vol_ratio"] > 1.5) & gan_dinh
    )

    return df


# ============ TÍNH ĐIỂM CẮT LỖ / CHỐT LỜI ============
def tinh_diem_cat_lo_chot_loi(row) -> tuple:
    entry = row["close"]
    stop_loss = min(row["day_cuc_bo"], row["low"]) if pd.notna(row["day_cuc_bo"]) else row["low"] * 0.97
    risk = entry - stop_loss
    take_profit = entry + risk * 2  # R:R = 1:2
    return round(entry, 2), round(stop_loss, 2), round(take_profit, 2)


# ============ GỬI EMAIL ============
def gui_email(noi_dung: str):
    email_user = os.environ["EMAIL_USER"]
    email_pass = os.environ["EMAIL_PASS"]
    email_to = os.environ["EMAIL_TO"]

    msg = MIMEText(noi_dung, "plain", "utf-8")
    msg["Subject"] = f"[Trade Bot] Tín hiệu {date.today()}"
    msg["From"] = email_user
    msg["To"] = email_to

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(email_user, email_pass)
        server.sendmail(email_user, [email_to], msg.as_string())


# ============ MAIN ============
def main():
    tin_hieu_ten = {
        "tin_hieu_nen_nen": "Nén nền cạn volume",
        "tin_hieu_test_ma": "Test nền / Test MA",
        "tin_hieu_breakout": "Breakout",
        "tin_hieu_churning": "Churning (phân phối tại đỉnh)",
    }

    ket_qua_email = []

    for ma in MA_CO_PHIEU:
        try:
            df = lay_du_lieu(ma)
            if df is None or len(df) < 25:
                continue
            df = tinh_cong_thuc(df)
            df = phat_hien_tin_hieu(df)

            row_moi_nhat = df.iloc[-1]

            for cot, ten in tin_hieu_ten.items():
                if bool(row_moi_nhat[cot]):
                    entry, sl, tp = tinh_diem_cat_lo_chot_loi(row_moi_nhat)
                    ket_qua_email.append(
                        f"{ma} — {ten}\n"
                        f"  Giá hiện tại: {entry}\n"
                        f"  Cắt lỗ: {sl}\n"
                        f"  Chốt lời: {tp}\n"
                    )
        except Exception as e:
            print(f"Lỗi mã {ma}: {e}")

    if ket_qua_email:
        noi_dung = "\n".join(ket_qua_email)
        noi_dung += "\n\n(Lưu ý: đây là tín hiệu tự động từ backtest đơn giản, không phải khuyến nghị đầu tư.)"
        gui_email(noi_dung)
        print("Đã gửi email với", len(ket_qua_email), "tín hiệu.")
    else:
        print("Không có tín hiệu nào lúc này.")


if __name__ == "__main__":
    main()
