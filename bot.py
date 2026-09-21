"""
BOT QUÉT TÍN HIỆU + GỬI EMAIL (bản tối ưu cho GitHub Actions)

Thay đổi so với bản gốc:
  1. Kiểm tra biến môi trường email NGAY từ đầu (fail-fast), tránh quét
     40-90 phút rồi mới báo thiếu secret.
  2. Bỏ sleep thừa trước khi kiểm tra VN-Index.
  3. Retry có backoff tăng dần + jitter thay vì sleep cố định.
  4. Chặn tín hiệu "ảo": nếu risk (entry - stop_loss) <= 0 thì bỏ qua mã đó.
  5. Dùng module logging có timestamp thay vì print thuần.
  6. Bọc main() trong try/except tổng — nếu lỗi giữa chừng vẫn có log rõ
     ràng thay vì job chết im lặng.
  7. QUAN TRỌNG NHẤT: file trang_thai_da_gui.json sẽ KHÔNG tồn tại giữa
     các lần chạy trên GitHub Actions (mỗi lần chạy là máy ảo mới) — xem
     phần "LƯU Ý VỀ TRẠNG THÁI" cuối file để biết cách khắc phục.
"""

from __future__ import annotations

import json
import logging
import os
import random
import smtplib
import time
from datetime import date, timedelta
from email.mime.text import MIMEText

import numpy as np
import pandas as pd
from vnstock import Listing, Quote

# ============ CẤU HÌNH ============
DS_BO_SUNG_THU_CONG = ["HHP"]

THOI_GIAN_NGHI_GIUA_MA = 6
THOI_GIAN_NGHI_KHI_BI_CHAN = 20
SO_LAN_THU_LAI = 2
FILE_TRANG_THAI = "trang_thai_da_gui.json"

HE_SO_ATR_CAT_LO = 1.5
TY_LE_RR = 2.0
KHOI_LUONG_TB_TOI_THIEU = 100_000
SO_YEU_TO_TOI_THIEU_DE_GUI = 2

NGAY_KET_THUC = date.today()
NGAY_BAT_DAU = NGAY_KET_THUC - timedelta(days=100)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bot")


# ============ KIỂM TRA CẤU HÌNH TRƯỚC KHI CHẠY ============
def kiem_tra_bien_moi_truong() -> tuple[str, str, str]:
    email_user = os.environ.get("EMAIL_USER", "").strip()
    email_pass = os.environ.get("EMAIL_PASS", "").strip()
    email_to = os.environ.get("EMAIL_TO", "").strip()
    thieu = [
        ten
        for ten, gia_tri in [
            ("EMAIL_USER", email_user),
            ("EMAIL_PASS", email_pass),
            ("EMAIL_TO", email_to),
        ]
        if not gia_tri
    ]
    if thieu:
        raise RuntimeError(
            f"Thiếu secret bắt buộc: {', '.join(thieu)}. "
            "Kiểm tra Settings > Secrets and variables > Actions trên GitHub."
        )
    return email_user, email_pass, email_to


# ============ LẤY DANH SÁCH MÃ ============
def lay_danh_sach_ma() -> list[str]:
    try:
        listing = Listing()
        try:
            ds = listing.symbols_by_exchange("HOSE")
            ds = list(ds["symbol"]) if hasattr(ds, "columns") else list(ds)
        except Exception:
            ds = listing.symbols_by_group("VN100")
        ds_day_du = list(dict.fromkeys(list(ds) + DS_BO_SUNG_THU_CONG))
        log.info(f"Lấy được {len(ds_day_du)} mã để quét (đã gộp thêm {DS_BO_SUNG_THU_CONG}).")
        return ds_day_du
    except Exception as e:
        log.warning(f"Không lấy được danh sách từ vnstock ({e}), dùng danh sách dự phòng.")
        return DS_BO_SUNG_THU_CONG + [
            "HPG", "MWG", "VCB", "VHM", "VIC", "GAS", "MSN", "TCB",
            "CTG", "BID", "VPB", "MBB", "ACB", "STB", "SSI", "VRE", "PLX", "POW",
            "GVR", "SAB", "HDB", "TPB", "BVH", "KDH", "PDR", "NVL", "DGC", "VJC",
        ]


# ============ ĐỌC / GHI TRẠNG THÁI ĐÃ GỬI ============
def doc_trang_thai_da_gui() -> dict:
    if os.path.exists(FILE_TRANG_THAI):
        try:
            with open(FILE_TRANG_THAI, "r", encoding="utf-8") as f:
                du_lieu = json.load(f)
        except Exception:
            du_lieu = {}
    else:
        du_lieu = {}
    return du_lieu.get(str(date.today()), {})


def ghi_trang_thai_da_gui(da_gui_hom_nay: dict) -> None:
    with open(FILE_TRANG_THAI, "w", encoding="utf-8") as f:
        json.dump({str(date.today()): da_gui_hom_nay}, f, ensure_ascii=False, indent=2)


# ============ LẤY DỮ LIỆU MỘT MÃ ============
def lay_du_lieu(ma: str) -> pd.DataFrame:
    q = Quote(symbol=ma, source="KBS")
    df = q.history(
        start=NGAY_BAT_DAU.strftime("%Y-%m-%d"),
        end=NGAY_KET_THUC.strftime("%Y-%m-%d"),
        interval="1D",
    )
    return df.reset_index(drop=True)


# ============ BỐI CẢNH THỊ TRƯỜNG (VN-INDEX) ============
def kiem_tra_thi_truong() -> tuple[bool, str]:
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
        log.warning(f"Lỗi kiểm tra VN-Index: {e}")
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


# ============ CHẤM ĐIỂM 4 YẾU TỐ ============
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
    return {"boi_canh_hop_le": boi_canh_hop_le, "yeu_to": yeu_to, "so_dat": so_dat}


def tinh_diem_cat_lo_chot_loi(row) -> tuple[float, float, float] | None:
    entry = row["close"]
    stop_loss_atr = entry - HE_SO_ATR_CAT_LO * row["atr14"] if pd.notna(row.get("atr14")) else None
    stop_loss_day = min(row["day_cuc_bo"], row["low"]) if pd.notna(row.get("day_cuc_bo")) else row["low"] * 0.97
    stop_loss = max(stop_loss_atr, stop_loss_day) if stop_loss_atr is not None else stop_loss_day

    risk = entry - stop_loss
    if risk <= 0:
        # Stop-loss tính ra >= giá vào lệnh -> tín hiệu không hợp lệ, bỏ qua.
        return None

    take_profit = entry + risk * TY_LE_RR
    return round(entry, 2), round(stop_loss, 2), round(take_profit, 2)


# ============ GỬI EMAIL ============
def gui_email(email_user: str, email_pass: str, email_to: str, noi_dung: str) -> None:
    msg = MIMEText(noi_dung, "plain", "utf-8")
    msg["Subject"] = f"[Trade Bot] Tín hiệu {date.today()}"
    msg["From"] = email_user
    msg["To"] = email_to

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(email_user, email_pass)
            server.sendmail(email_user, [email_to], msg.as_string())
    except smtplib.SMTPAuthenticationError as e:
        raise RuntimeError("Gmail từ chối đăng nhập. Kiểm tra lại EMAIL_USER và EMAIL_PASS.") from e


# ============ QUÉT MỘT MÃ (kèm retry có backoff) ============
def quet_mot_ma(ma: str, da_gui_hom_nay: dict) -> str | None:
    """Trả về khối nội dung email nếu có tín hiệu mới, ngược lại None."""
    for lan_thu in range(SO_LAN_THU_LAI):
        try:
            time.sleep(THOI_GIAN_NGHI_GIUA_MA + random.uniform(0, 1.5))
            df = lay_du_lieu(ma)
            if df is None or len(df) < 25:
                return None

            df = tinh_cong_thuc(df)
            row = df.iloc[-1]
            danh_gia = danh_gia_churning(row)

            if not (danh_gia["boi_canh_hop_le"] and danh_gia["so_dat"] >= SO_YEU_TO_TOI_THIEU_DE_GUI):
                return None

            khoa = f"{ma}_churning"
            if khoa in da_gui_hom_nay:
                log.info(f"Bỏ qua {ma} (đã gửi hôm nay rồi)")
                return None

            diem = tinh_diem_cat_lo_chot_loi(row)
            if diem is None:
                log.info(f"Bỏ qua {ma}: risk tính ra <= 0, tín hiệu không hợp lệ.")
                return None
            entry, sl, tp = diem

            dat = [k for k, v in danh_gia["yeu_to"].items() if v]
            thieu = [k for k, v in danh_gia["yeu_to"].items() if not v]
            rsi_str = f"{row['rsi14']:.1f}" if pd.notna(row.get("rsi14")) else "N/A"

            da_gui_hom_nay[khoa] = True
            return (
                f"{ma} — Churning ({danh_gia['so_dat']}/4 yếu tố)\n"
                f"  Giá hiện tại: {entry}\n"
                f"  RSI14: {rsi_str}\n"
                f"  Cắt lỗ: {sl}\n"
                f"  Chốt lời: {tp}\n"
                f"  Đạt: {', '.join(dat) if dat else '(không có)'}\n"
                f"  Thiếu: {', '.join(thieu) if thieu else '(không thiếu gì — đủ 4/4)'}\n"
            )
        except Exception as e:
            log.warning(f"Lỗi mã {ma} (lần {lan_thu + 1}/{SO_LAN_THU_LAI}): {e}")
            if "limit" in str(e).lower() or "rate" in str(e).lower():
                cho = THOI_GIAN_NGHI_KHI_BI_CHAN * (lan_thu + 1) + random.uniform(0, 3)
                log.info(f"Bị giới hạn API, đợi {cho:.0f} giây rồi thử lại...")
                time.sleep(cho)
            else:
                return None
    return None


def main() -> None:
    email_user, email_pass, email_to = kiem_tra_bien_moi_truong()

    danh_sach_ma = lay_danh_sach_ma()
    da_gui_hom_nay = doc_trang_thai_da_gui()

    thi_truong_thuan_loi, ghi_chu_thi_truong = kiem_tra_thi_truong()
    log.info(f"Bối cảnh thị trường: {ghi_chu_thi_truong}")
    if not thi_truong_thuan_loi:
        log.info("VN-Index đang dưới MA20 — tạm ngưng gửi tín hiệu mua mới.")
        return

    ket_qua_email = []
    for idx, ma in enumerate(danh_sach_ma):
        khoi = quet_mot_ma(ma, da_gui_hom_nay)
        if khoi:
            ket_qua_email.append(khoi)
        if (idx + 1) % 50 == 0:
            log.info(f"Đã quét {idx + 1}/{len(danh_sach_ma)} mã...")

    if not ket_qua_email:
        log.info("Không có tín hiệu mới lúc này.")
        return

    noi_dung = f"{ghi_chu_thi_truong}\n\n" + "\n".join(ket_qua_email)
    noi_dung += (
        "\n\n(Lưu ý: đây là tín hiệu tự động, không phải khuyến nghị đầu tư. "
        "Mã đạt càng nhiều yếu tố (gần 4/4) thì mức độ tin cậy theo backtest càng cao. "
        "Luôn tự kiểm tra lại biểu đồ, đặc biệt các yếu tố còn thiếu, trước khi quyết định.)"
    )
    try:
        gui_email(email_user, email_pass, email_to, noi_dung)
        log.info(f"Đã gửi email với {len(ket_qua_email)} tín hiệu mới.")
        ghi_trang_thai_da_gui(da_gui_hom_nay)
    except Exception as e:
        log.error(f"Gửi email thất bại: {e}. Sẽ tự động thử gửi lại ở lần chạy kế tiếp.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"Bot dừng vì lỗi không xử lý được: {e}")
        raise


# ============ LƯU Ý VỀ TRẠNG THÁI TRÊN GITHUB ACTIONS ============
# `trang_thai_da_gui.json` chỉ tồn tại trong máy ảo của LẦN CHẠY HIỆN TẠI.
# Nếu workflow chạy nhiều lần/ngày, bot sẽ không biết đã gửi mã nào rồi và
# có thể gửi trùng email. Hai cách khắc phục phổ biến, chọn 1:
#
# 1) Commit file trạng thái ngược lại repo ở cuối mỗi lần chạy (đơn giản
#    nhất). Thêm vào cuối file .yml workflow, sau bước chạy script:
#
#      - name: Lưu trạng thái đã gửi
#        run: |
#          git config user.name "trade-bot"
#          git config user.email "bot@users.noreply.github.com"
#          git add trang_thai_da_gui.json
#          git diff --staged --quiet || git commit -m "cap nhat trang thai gui"
#          git push
#
# 2) Dùng actions/cache để lưu file giữa các lần chạy (không tạo commit
#    rác vào repo):
#
#      - uses: actions/cache@v4
#        with:
#          path: trang_thai_da_gui.json
#          key: trang-thai-gui-${{ github.run_id }}
#          restore-keys: trang-thai-gui-
#
# Nếu bot chỉ chạy đúng 1 lần/ngày thì vấn đề này không ảnh hưởng gì, có
# thể bỏ qua.
