"""
BOT QUÉT TÍN HIỆU + GỬI EMAIL
Bản cập nhật: CHỈ gửi email khi có tín hiệu "Churning" (kỳ vọng lợi nhuận
tốt nhất theo backtest +0.46R). Test MA/Breakout vẫn được quét và ghi log
(để theo dõi/backtest thêm sau này) nhưng KHÔNG gửi email — vì backtest 2 năm
cho thấy 2 loại này gần như hòa vốn, gửi liên tục sẽ làm loãng giá trị email.
"""

from datetime import date, timedelta
import os
import time
import json
import smtplib
from email.mime.text import MIMEText

import numpy as np
import pandas as pd

from vnstock import Quote

# ============ CẤU HÌNH ============
MA_CO_PHIEU = [
    "TIP", "MCP", "HPG", "MWG", "VCB", "VHM", "VIC", "GAS", "MSN", "TCB",
    "CTG", "BID", "VPB", "MBB", "ACB", "STB", "SSI", "VRE", "PLX", "POW",
    "GVR", "SAB", "HDB", "TPB", "BVH", "KDH", "PDR", "NVL", "DGC", "VJC",
]

THOI_GIAN_NGHI_GIUA_MA = 6
THOI_GIAN_NGHI_KHI_BI_CHAN = 20
SO_LAN_THU_LAI = 2
FILE_TRANG_THAI = "trang_thai_da_gui.json"

HE_SO_ATR_CAT_LO = 1.5
TY_LE_RR = 2.0

KHOI_LUONG_TB_TOI_THIEU = 100_000

NGAY_KET_THUC = date.today()
NGAY_BAT_DAU = NGAY_KET_THUC - timedelta(days=100)


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


# ============ TÍNH RSI (chuẩn Wilder, dùng EWM) ============
def tinh_rsi(close: pd.Series, chu_ky: int = 14) -> pd.Series:
    thay_doi = close.diff()
    tang = thay_doi.clip(lower=0)
    giam = -thay_doi.clip(upper=0)
    tb_tang = tang.ewm(alpha=1 / chu_ky, adjust=False, min_periods=chu_ky).mean()
    tb_giam = giam.ewm(alpha=1 / chu_ky, adjust=False, min_periods=chu_ky).mean()
    rs = tb_tang / tb_giam.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ============ TÍNH ATR (chuẩn Wilder, dùng EWM) ============
def tinh_atr(df: pd.DataFrame, chu_ky: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close_truoc = (df["high"] - df["close"].shift(1)).abs()
    low_close_truoc = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close_truoc, low_close_truoc], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / chu_ky, adjust=False, min_periods=chu_ky).mean()
    return atr


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

    df["rsi14"] = tinh_rsi(df["close"], 14)
    df["atr14"] = tinh_atr(df, 14)

    return df


# ============ PHÁT HIỆN TÍN HIỆU ============
# Vẫn tính cả test_ma/breakout để ghi log theo dõi, nhưng chỉ churning được gửi email
def phat_hien_tin_hieu(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    thanh_khoan_du = df["vol_ma20"] >= KHOI_LUONG_TB_TOI_THIEU

    than_nho = df["than_pct"] < 40
    bong_duoi_dai = df["bong_duoi_pct"] > 50
    gan_ma = (df["close"] - df["ma20"]).abs() / df["ma20"] <= 0.02
    rsi_khong_qua_ban = df["rsi14"] > 35
    df["tin_hieu_test_ma"] = (
        (than_nho | bong_duoi_dai) & (df["vol_ratio"].between(0.5, 0.7)) & gan_ma
        & rsi_khong_qua_ban & thanh_khoan_du
    )

    than_lon = df["than_pct"] > 70
    dong_cua_cao = (df["high"] - df["close"]) / df["bien_do"].replace(0, np.nan) <= 0.10
    vuot_dinh = df["close"] > df["dinh_cuc_bo"]
    vuot_ma = df["close"] > df["ma20"]
    vol_no = df["vol_ratio"] > 2.0
    rsi_khong_qua_mua = df["rsi14"] < 78
    df["tin_hieu_breakout"] = (
        than_lon & dong_cua_cao & vuot_dinh & vuot_ma & vol_no & rsi_khong_qua_mua & thanh_khoan_du
    )

    gan_dinh = (df["close"] - df["dinh_cuc_bo"]).abs() / df["dinh_cuc_bo"] <= 0.03
    rsi_qua_mua = df["rsi14"] > 70
    df["tin_hieu_churning"] = (
        (df["than_pct"] < 30) & (df["bong_tren_pct"] > 40) & (df["vol_ratio"] > 1.5)
        & gan_dinh & rsi_qua_mua & thanh_khoan_du
    )

    return df


# ============ TÍNH ĐIỂM CẮT LỖ / CHỐT LỜI (dùng ATR, có dự phòng) ============
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

    if stop_loss_atr is not None:
        stop_loss = max(stop_loss_atr, stop_loss_day)
    else:
        stop_loss = stop_loss_day

    risk = entry - stop_loss
    take_profit = entry + risk * TY_LE_RR
    return round(entry, 2), round(stop_loss, 2), round(take_profit, 2)


# ============ GỬI EMAIL ============
def gui_email(noi_dung: str):
    email_user = os.environ.get("EMAIL_USER", "").strip()
    email_pass = os.environ.get("EMAIL_PASS", "").strip()
    email_to = os.environ.get("EMAIL_TO", "").strip()

    if not email_user or not email_pass or not email_to:
        raise RuntimeError("Thiếu secret: kiểm tra EMAIL_USER, EMAIL_PASS, EMAIL_TO.")

    msg = MIMEText(noi_dung, "plain", "utf-8")
    msg["Subject"] = f"[Trade Bot] ⭐ Churning {date.today()}"
    msg["From"] = email_user
    msg["To"] = email_to

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(email_user, email_pass)
            server.sendmail(email_user, [email_to], msg.as_string())
    except smtplib.SMTPAuthenticationError:
        raise RuntimeError(
            "Gmail từ chối đăng nhập. Kiểm tra lại EMAIL_USER và EMAIL_PASS (App Password)."
        )


# ============ MAIN ============
def main():
    da_gui_hom_nay = doc_trang_thai_da_gui()

    time.sleep(THOI_GIAN_NGHI_GIUA_MA)
    thi_truong_thuan_loi, ghi_chu_thi_truong = kiem_tra_thi_truong()
    print(f"Bối cảnh thị trường: {ghi_chu_thi_truong}")

    if not thi_truong_thuan_loi:
        print("VN-Index đang dưới MA20 — tạm ngưng gửi tín hiệu mua mới.")
        return

    ket_qua_email = []
    co_tin_hieu_moi = False

    for ma in MA_CO_PHIEU:
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
                df = phat_hien_tin_hieu(df)

                row_moi_nhat = df.iloc[-1]
                rsi_hien_tai = row_moi_nhat.get("rsi14")
                rsi_str = f"{rsi_hien_tai:.1f}" if pd.notna(rsi_hien_tai) else "N/A"

                # Ghi log tất cả tín hiệu (kể cả test_ma, breakout) để theo dõi/backtest sau
                if bool(row_moi_nhat["tin_hieu_test_ma"]):
                    print(f"[Chỉ ghi log, không gửi mail] {ma} - Test MA")
                if bool(row_moi_nhat["tin_hieu_breakout"]):
                    print(f"[Chỉ ghi log, không gửi mail] {ma} - Breakout")

                # CHỈ gửi email cho tín hiệu Churning
                if bool(row_moi_nhat["tin_hieu_churning"]):
                    khoa = f"{ma}_tin_hieu_churning"
                    if khoa in da_gui_hom_nay:
                        print(f"Bỏ qua {ma} - Churning (đã gửi hôm nay rồi)")
                    else:
                        entry, sl, tp = tinh_diem_cat_lo_chot_loi(row_moi_nhat)
                        ket_qua_email.append(
                            f"{ma} — ⭐ Churning (phân phối tại đỉnh)\n"
                            f"  Giá hiện tại: {entry}\n"
                            f"  RSI14: {rsi_str}\n"
                            f"  Cắt lỗ: {sl}\n"
                            f"  Chốt lời: {tp}\n"
                        )
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

    if ket_qua_email:
        noi_dung = f"{ghi_chu_thi_truong}\n\n" + "\n".join(ket_qua_email)
        noi_dung += (
            "\n\n(Lưu ý: đây là tín hiệu tự động từ backtest đơn giản, không phải "
            "khuyến nghị đầu tư. Theo backtest 2 năm, Churning là loại tín hiệu duy nhất "
            "có kỳ vọng lợi nhuận rõ ràng (+0.46R) — vì vậy bot chỉ gửi email cho loại này, "
            "các tín hiệu Test MA/Breakout khác được ghi log nhưng không gửi mail.)"
        )
        try:
            gui_email(noi_dung)
            print("Đã gửi email với", len(ket_qua_email), "tín hiệu Churning mới.")
            if co_tin_hieu_moi:
                ghi_trang_thai_da_gui(da_gui_hom_nay)
        except Exception as e:
            print(f"Gửi email thất bại: {e}")
            print("Sẽ tự động thử gửi lại ở lần chạy kế tiếp.")
    else:
        print("Không có tín hiệu Churning mới lúc này (bình thường — đây là tín hiệu hiếm).")


if __name__ == "__main__":
    main()
