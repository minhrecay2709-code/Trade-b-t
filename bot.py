"""
BOT QUÉT TÍN HIỆU + GỬI EMAIL
Bản đầy đủ: lọc VN-Index + chống gửi trùng lặp + RSI + ATR (cắt lỗ thích ứng)
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
    "FPT", "VNM", "HPG", "MWG", "VCB", "VHM", "VIC", "GAS", "MSN", "TCB",
    "CTG", "BID", "VPB", "MBB", "ACB", "STB", "SSI", "VRE", "PLX", "POW",
    "GVR", "SAB", "HDB", "TPB", "BVH", "KDH", "PDR", "NVL", "DGC", "VJC",
]

THOI_GIAN_NGHI_GIUA_MA = 6
THOI_GIAN_NGHI_KHI_BI_CHAN = 30
SO_LAN_THU_LAI = 3
FILE_TRANG_THAI = "trang_thai_da_gui.json"

HE_SO_ATR_CAT_LO = 1.5  # cắt lỗ = giá vào lệnh - 1.5 x ATR14
TY_LE_RR = 2.0           # chốt lời = rủi ro x 2 (R:R 1:2)

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


# ============ KIỂM TRA BỐI CẢNH T
