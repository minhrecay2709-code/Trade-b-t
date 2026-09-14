"""
BACKTEST MỞ RỘNG: nhiều mã hơn (tự động lấy danh sách VN100) + kiểm tra
xem việc chấm điểm 4 yếu tố của Churning (đạt 2/4, 3/4, hay 4/4) có tạo
ra sự khác biệt kết quả theo đúng thứ tự tăng dần hay không.
Dùng cách đo ĐÚNG: mô phỏng chạm SL hay TP trước (không phải so giá đơn thuần).
"""

from datetime import date, timedelta
import time
import numpy as np
import pandas as pd
from vnstock import Quote, Listing

# ============ CẤU HÌNH ============
SO_NGAY_LICH_SU = 1095  # 3 năm (tăng từ 2 năm)
THOI_GIAN_NGHI = 6
KHOI_LUONG_TB_TOI_THIEU = 100_000
HE_SO_ATR_CAT_LO = 1.5
TY_LE_RR = 2.0
SO_PHIEN_TOI_DA_NAM_GIU = 15

NGAY_KET_THUC = date.today()
NGAY_BAT_DAU = NGAY_KET_THUC - timedelta(days=SO_NGAY_LICH_SU)


def lay_danh_sach_ma():
    """Lấy danh sách mã tự động (VN100) thay vì gõ tay — tăng cỡ mẫu."""
    ds_them = ["TIP", "MCP"]  # 2 mã bạn đang đầu tư, luôn giữ lại
    try:
        listing = Listing()
        ds_vn100 = listing.symbols_by_group("VN100")
        ds = list(dict.fromkeys(ds_them + list(ds_vn100)))
        print(f"Lấy được {len(ds)} mã từ VN100 (đã gộp thêm {ds_them}).")
        return ds
    except Exception as e:
        print(f"Không lấy được danh sách VN100 ({e}), dùng danh sách dự phòng 30 mã.")
        return ds_them + [
            "HPG", "MWG", "VCB", "VHM", "VIC", "GAS", "MSN", "TCB",
            "CTG", "BID", "VPB", "MBB", "ACB", "STB", "SSI", "VRE", "PLX", "POW",
            "GVR", "SAB", "HDB", "TPB", "BVH", "KDH", "PDR", "NVL", "DGC", "VJC",
        ]


def lay_du_lieu(ma):
    q = Quote(symbol=ma, source="KBS")
    df = q.history(
        start=NGAY_BAT_DAU.strftime("%Y-%m-%d"),
        end=NGAY_KET_THUC.strftime("%Y-%m-%d"),
        interval="1D",
    )
    return df.reset_index(drop=True)


def tinh_rsi(close, chu_ky=14):
    thay_doi = close.diff()
    tang = thay_doi.clip(lower=0)
    giam = -thay_doi.clip(upper=0)
    tb_tang = tang.ewm(alpha=1 / chu_ky, adjust=False, min_periods=chu_ky).mean()
    tb_giam = giam.ewm(alpha=1 / chu_ky, adjust=False, min_periods=chu_ky).mean()
    rs = tb_tang / tb_giam.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def tinh_atr(df, chu_ky=14):
    high_low = df["high"] - df["low"]
    high_close_truoc = (df["high"] - df["close"].shift(1)).abs()
    low_close_truoc = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close_truoc, low_close_truoc], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / chu_ky, adjust=False, min_periods=chu_ky).mean()


def tinh_cong_thuc(df):
    df = df.copy()
    df["bien_do"] = df["high"] - df["low"]
    df["than_pct"] = (df["close"] - df["open"]).abs() / df["bien_do"].replace(0, np.nan) * 100
    df["bong_tren_pct"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["bien_do"].replace(0, np.nan) * 100
    df["bong_duoi_pct"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["bien_do"].replace(0, np.nan) * 100

    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]
    df["ma20"] = df["close"].rolling(20).mean()
    df["dinh_cuc_bo"] = df["high"].shift(1).rolling(20).max()
    df["day_cuc_bo"] = df["low"].shift(1).rolling(20).min()
    df["rsi14"] = tinh_rsi(df["close"], 14)
    df["atr14"] = tinh_atr(df, 14)
    return df


# ============ 4 YẾU TỐ CỦA CHURNING (bối cảnh "gần đỉnh" là điều kiện nền, luôn bắt buộc) ============
def tinh_cac_yeu_to_churning(df):
    df = df.copy()
    thanh_khoan_du = df["vol_ma20"] >= KHOI_LUONG_TB_TOI_THIEU
    gan_dinh = (df["close"] - df["dinh_cuc_bo"]).abs() / df["dinh_cuc_bo"] <= 0.03

    # Bối cảnh nền bắt buộc: phải đang ở gần đỉnh cục bộ + đủ thanh khoản
    df["boi_canh_hop_le"] = gan_dinh & thanh_khoan_du & df["dinh_cuc_bo"].notna()

    # 4 yếu tố được chấm điểm (mỗi yếu tố đúng/sai độc lập)
    df["yt_than_nen_nho"] = df["than_pct"] < 30
    df["yt_bong_tren_dai"] = df["bong_tren_pct"] > 40
    df["yt_volume_dot_bien"] = df["vol_ratio"] > 1.5
    df["yt_rsi_qua_mua"] = df["rsi14"] > 70

    df["so_yeu_to_dat"] = (
        df["yt_than_nen_nho"].astype(int)
        + df["yt_bong_tren_dai"].astype(int)
        + df["yt_volume_dot_bien"].astype(int)
        + df["yt_rsi_qua_mua"].astype(int)
    )
    return df


def tinh_diem_cat_lo_chot_loi(row):
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
    return entry, stop_loss, take_profit


def mo_phong_giao_dich(df, i, sl, tp, max_ngay):
    for offset in range(1, max_ngay + 1):
        idx = i + offset
        if idx >= len(df):
            return "chua_dong"
        gia_thap = df["low"].iloc[idx]
        gia_cao = df["high"].iloc[idx]
        if gia_thap <= sl:
            return "thua"
        if gia_cao >= tp:
            return "thang"
    return "chua_dong"


def main():
    danh_sach_ma = lay_danh_sach_ma()
    print(f"\nBacktest tren {len(danh_sach_ma)} ma, {SO_NGAY_LICH_SU} ngay lich su\n")

    ket_qua_theo_so_yeu_to = {2: [], 3: [], 4: []}

    for idx_ma, ma in enumerate(danh_sach_ma):
        print(f"[{idx_ma+1}/{len(danh_sach_ma)}] Dang xu ly {ma}...")
        try:
            time.sleep(THOI_GIAN_NGHI)
            df = lay_du_lieu(ma)
            if df is None or len(df) < 30:
                continue
            df = tinh_cong_thuc(df)
            df = tinh_cac_yeu_to_churning(df)

            for i in range(20, len(df) - SO_PHIEN_TOI_DA_NAM_GIU):
                row = df.iloc[i]
                if not bool(row["boi_canh_hop_le"]):
                    continue
                so_yt = int(row["so_yeu_to_dat"])
                if so_yt < 2:
                    continue
                entry, sl, tp = tinh_diem_cat_lo_chot_loi(row)
                ket_qua = mo_phong_giao_dich(df, i, sl, tp, SO_PHIEN_TOI_DA_NAM_GIU)
                ket_qua_theo_so_yeu_to[so_yt].append(ket_qua)
        except Exception as e:
            print(f"Loi ma {ma}: {e}")
            if "limit" in str(e).lower() or "rate" in str(e).lower():
                print("Bi gioi han API, doi 30 giay...")
                time.sleep(30)

    print("\n\n========== KET QUA THEO SO YEU TO DAT DUOC (2/4, 3/4, 4/4) ==========\n")
    for so_yt in [2, 3, 4]:
        ds = ket_qua_theo_so_yeu_to[so_yt]
        tong = len(ds)
        print(f"--- Dat {so_yt}/4 yeu to ---")
        print(f"Tong so lan: {tong}")
        if tong == 0:
            print("(khong co du lieu)\n")
            continue
        so_thang = ds.count("thang")
        so_thua = ds.count("thua")
        so_chua_dong = ds.count("chua_dong")
        da_dong = so_thang + so_thua
        print(f"Thang: {so_thang} ({so_thang/tong*100:.1f}%)")
        print(f"Thua: {so_thua} ({so_thua/tong*100:.1f}%)")
        print(f"Chua dong: {so_chua_dong} ({so_chua_dong/tong*100:.1f}%)")
        if da_dong > 0:
            ky_vong_R = (so_thang * TY_LE_RR - so_thua * 1) / da_dong
            print(f"Ky vong loi nhuan (R): {ky_vong_R:+.2f}R")
        print()

    print("========== GHI CHU DOC KET QUA ==========")
    print("- Neu ky vong R tang dan theo dung thu tu 2/4 < 3/4 < 4/4,")
    print("  he thong cham diem 4 yeu to CO Y NGHIA, nen dung.")
    print("- Neu khong theo thu tu do (vd 2/4 lai tot hon 4/4), he thong")
    print("  cham diem KHONG dang tin, nen giu nguyen kieu 'dat du 4/4 moi gui'.")
    print("- So luong '2/4' qua it hoac qua nhieu so voi '4/4' cung can luu y")
    print("  khi danh gia do tin cay cua so sanh.")


main()
