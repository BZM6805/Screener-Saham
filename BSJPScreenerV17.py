#!/usr/bin/env python3
"""
BSJPScreenerV16_TeleBot.py
- UPGRADED: Imports FTScreenerV8.
- NEW FEATURE: Added BSJP (Beli Sore Jual Pagi) calculation using real-time data.
- FIXED: Robustly handles NaN/empty values for Buy/TP Range, Target, and SL to prevent empty output.
- FOCUS: Highlights Buy/TP Range, Target, SL, and RR Ratio.
- Builds on BSJPScreenerV15_TeleBot.py
Place this file alongside BSJPScreenerV10.py, FTScreenerV8.py, HarmonicScreener.py and JPScreenerV3html.py
Run: python BSJPScreenerV16_TeleBot.py
"""
import os, sys, time, json, traceback, requests
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np 
import yfinance as yf # <<< DITAMBAHKAN untuk BSJP real-time

# ----------------- Config / imports -----------------
try:
    from BSJPScreenerV10 import BOT_TOKEN, CHAT_IDS
except Exception as e:
    print('[BOOT] Failed to import BSJPScreenerV10:', e)
    BOT_TOKEN = None
    CHAT_IDS = []

if not BOT_TOKEN:
    print('[BOT] BOT_TOKEN not found in BSJPScreenerV10.py. Please add BOT_TOKEN and CHAT_IDS.')
    sys.exit(1)

try:
    # Use V8 which includes liquidity filters and trading levels
    import FTScreenerV8 as fts
except Exception as e:
    print('[BOOT] Failed to import FTScreenerV8:', e)
    fts = None

# Harmonic module (optional)
try:
    import HarmonicScreener as harm_mod
except Exception:
    harm_mod = None

# --- IMPORT JPScreener modules ---
try:
    import JPScreenerV2 as jps
except Exception as e:
    print('[BOOT] Failed to import JPScreenerV2.py:', e)
    jps = None

try:
    import JPScreenerV3html as jps_v3
except Exception as e:
    print('[BOOT] Failed to import JPScreenerV3html.py (v3.2):', e)
    jps_v3 = None
# --------------------------------

# Paths
BASE_DIR = Path.cwd()
OUTPUT_DIR = BASE_DIR / 'fts_output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HARM_DIR = BASE_DIR / 'harmonic_eod_results'
HARM_FILE = HARM_DIR / 'harmonic_bandar.csv'

TELE_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
# --- UPDATE OFFSET FILE TO V16 ---
OFFSET_FILE = BASE_DIR / 'tele_offset_v16.txt'
POLL_INTERVAL = 5


# Hasil screening akan disimpan di sini setelah dijalankan pertama kali
GLOBAL_SCREENER_RESULTS = None 

# --------------- Screener runner & caching -------------------
def run_screener_if_needed(force=False):
    """Jalankan screener FTS + JPScreener V3.2 (cached). If force=True, re-run."""
    global GLOBAL_SCREENER_RESULTS
    if GLOBAL_SCREENER_RESULTS is None or force:
        # Initialize / re-run
        print('[SCREENER] Starting screener run (force=%s)...' % force)
        GLOBAL_SCREENER_RESULTS = {}
        # Run main FTScreener first (if available)
        if fts:
            try:
                print('[SCREENER] Running FTScreenerV8.run_full_scan() ...')
                GLOBAL_SCREENER_RESULTS = fts.run_full_scan() or {}
            except Exception as e:
                print('[SCREENER] FTScreenerV8.run_full_scan failed:', e, traceback.format_exc())
                GLOBAL_SCREENER_RESULTS = {}
        else:
            print('[SCREENER] FTScreenerV8 not available. Starting with empty results.')
            GLOBAL_SCREENER_RESULTS = {}

        # Try to run JPScreener V3.2 (preferred)
        if jps_v3:
            try:
                print('[SCREENER] Running JPScreenerV3html.run_all_jp_screeners_v3_2() ...')
                jp_results = jps_v3.run_all_jp_screeners_v3_2()
                # jp_results uses keys: 'ichimoku','heikenashi','renko','continuation','reversal','combined'
                # We'll map them into GLOBAL_SCREENER_RESULTS with a 'jp_' prefix to avoid collisions
                for k, v in (jp_results or {}).items():
                    GLOBAL_SCREENER_RESULTS[f'jp_{k}'] = v
                print('[SCREENER] JPScreenerV3.2 completed.')
            except Exception as e:
                print(f"[SCREENER ERROR] JPScreenerV3.2 failed: {e}", traceback.format_exc())
                # fallback: try older jps if present
                if jps:
                    try:
                        jp_results_old = jps.run_all_jp_screeners()
                        GLOBAL_SCREENER_RESULTS.update(jp_results_old)
                        print('[SCREENER] Fallback JPScreenerV2 completed.')
                    except Exception as e2:
                        print('[SCREENER ERROR] Fallback JPScreenerV2 failed:', e2, traceback.format_exc())
        else:
            # if v3 not available, fallback to older jps module if present
            if jps:
                try:
                    jp_results_old = jps.run_all_jp_screeners()
                    GLOBAL_SCREENER_RESULTS.update(jp_results_old)
                    print('[SCREENER] JPScreenerV2 completed.')
                except Exception as e:
                    print('[SCREENER ERROR] JPScreenerV2 failed:', e, traceback.format_exc())
    return GLOBAL_SCREENER_RESULTS

# --------------- BSJP Core Functions (NEW) -------------------

def fetch_realtime(ticker_list):
    """Mengambil harga real-time (last price) dari yfinance."""
    data = {}
    # yfinance requires .JK suffix for IDX stocks
    full_ticker_list = [t + ".JK" for t in ticker_list]
    # Set timeout for yfinance download
    try:
        yf_data = yf.download(full_ticker_list, period="2d", interval="1d", progress=False, timeout=10)
    except Exception as e:
        print(f"[BSJP] yfinance download failed: {e}")
        return data

    if yf_data.empty: return data
    
    for i, t_jk in enumerate(full_ticker_list):
        t = ticker_list[i]
        if len(ticker_list) == 1: info = yf_data
        else:
            if 'Close' in yf_data.columns: 
                # Handle multi-index columns for multiple tickers
                if t_jk in yf_data.columns.get_level_values(1):
                     info = yf_data.xs(t_jk, level=1, axis=1)
                else: continue
            else: continue
            
        if not info.empty and 'Close' in info.columns and not info['Close'].empty:
            # Mengambil harga penutupan terakhir (yang merupakan harga real-time)
            last = info.iloc[-1]
            data[t] = {
                'rt_price': float(last['Close']),
                'rt_volume': float(last['Volume'])
            }
    return data

def run_bsjp_calculation(fast_df):
    """
    Menghitung skor BSJP berdasarkan hasil Fast Trading (EOD data)
    dan harga real-time (via yfinance).
    Asumsi kolom FastScore, EMA13, SMA20, EMA8, vol_sma20, val_sma20 ada di fast_df.
    """
    if fast_df.empty or 'FastScore' not in fast_df.columns:
        return pd.DataFrame()
        
    # Filter saham yang memenuhi kriteria likuiditas (val_sma20 > 1M)
    # Gunakan .get() untuk keamanan jika kolom tidak ada
    liquid_tickers = fast_df[fast_df.get('val_sma20', 0) > 1_000_000]['ticker'].tolist() 
    # Hanya ambil top 50 FastScore untuk di-fetch
    top_tickers_fast = fast_df.sort_values('FastScore', ascending=False).head(50)['ticker'].tolist()
    tickers_to_fetch = list(set(liquid_tickers) & set(top_tickers_fast))

    if not tickers_to_fetch:
        return pd.DataFrame()

    rt = fetch_realtime(tickers_to_fetch)
        
    df = fast_df[fast_df['ticker'].isin(rt.keys())].copy().reset_index(drop=True)
    df['rt_price'] = df['ticker'].apply(lambda x: rt.get(x,{}).get('rt_price', np.nan))
    df = df.dropna(subset=['rt_price', 'close'])
    
    if df.empty: return pd.DataFrame()
    
    df['gap_up_prob'] = ((df['rt_price'] - df['close']) / df['close']).fillna(0)
    
    # Hitung Evening Strength (asumsi kolom teknikal ada)
    df['evening_strength'] = (
        (df.get('EMA13', 0) > df.get('SMA20', 0)).astype(int) +
        (df.get('volume', 0) > df.get('vol_sma20', 1)).astype(int) +
        (df.get('close', 0) > df.get('EMA8', 0)).astype(int)
    )
    # Rumus BSJP Score
    df['SorePagiScore'] = (df['gap_up_prob'] * 100 * 0.4) + (df['evening_strength'] * 0.4) + (df['FastScore'] / 100 * 0.2)
    
    # Pilih kolom yang relevan
    cols = ['ticker','close','rt_price','gap_up_prob','evening_strength','FastScore','SorePagiScore']
    cols = [c for c in cols if c in df.columns]
    
    return df.sort_values('SorePagiScore', ascending=False).head(10)[cols]

# --------------- Telegram helpers -------------------
# (Tidak ada perubahan pada fungsi helper Telegram: load_offset, send_long_message, save_offset, get_updates, send_document)
def load_offset():
    try:
        if OFFSET_FILE.exists():
            return int(OFFSET_FILE.read_text().strip())
    except Exception:
        pass
    return 0

def send_long_message(chat_id, text, parse_mode='Markdown'):
    # Telegram limit 4096 chars
    limit = 3800  # aman
    
    # Jika text kurang dari limit, kirim langsung
    if len(text) <= limit:
         url = f"{TELE_API}/sendMessage"
         payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode, "disable_web_page_preview": True}
         try:
            r = requests.post(url, data=payload, timeout=15)
            if r.status_code != 200:
                print('[TG] send_long_message failed', r.status_code, r.text)
            return r.status_code == 200
         except Exception as e:
            print('[TG] send_long_message exception:', e)
            return False

    # Jika text melebihi limit, bagi-bagi
    parts = []
    while len(text) > 0:
        cut = text[:limit]
        parts.append(cut)
        text = text[limit:]

    for p in parts:
        ok = send_long_message(chat_id, p, parse_mode=parse_mode)
        time.sleep(0.5)
    return True


def save_offset(offset):
    try:
        OFFSET_FILE.write_text(str(offset))
    except Exception:
        pass


def get_updates(offset=None, timeout=30):
    url = f"{TELE_API}/getUpdates"
    params = {"timeout": timeout, "allowed_updates": json.dumps(["message"]) }
    if offset:
        params['offset'] = offset
    try:
        r = requests.get(url, params=params, timeout=timeout+10)
        if r.status_code == 200:
            return r.json().get('result', [])
    except Exception as e:
        print('[TG] get_updates error:', e)
    return []


def send_document(chat_id, filename, data_bytes, caption=None):
    url = f"{TELE_API}/sendDocument"
    files = {'document': (filename, data_bytes)}
    payload = {'chat_id': chat_id}
    if caption:
        payload['caption'] = caption
    try:
        r = requests.post(url, data=payload, files=files, timeout=30)
        if r.status_code != 200:
            print('[TG] sendDocument failed', r.status_code, r.text)
            return False
        return True
    except Exception as e:
        print('[TG] sendDocument exception:', e)
        return False

# ---------------- Formatting helpers ----------------
def fmt_int(x):
    try:
        # Converts NaN/None to 0, formats with thousands separator
        val = int(x) if pd.notna(x) and pd.to_numeric(x, errors='coerce') > 0 else 0
        return f"{val:,}"
    except Exception:
        return '?' # Fallback to '?'

def format_rr(rr):
    try:
        rr_ratio = float(rr)
        return f"R/R: {rr_ratio:.2f}" if pd.notna(rr_ratio) and rr_ratio > 0 else "R/R: N/A"
    except Exception:
        return "R/R: N/A"

# Helper to get string value, handling potential NaN/None
def get_str(r, key):
    val = r.get(key)
    # Check if value is NaN, None, or empty string. If so, return '?'
    if pd.isna(val) or val is None or str(val).strip() == '':
        return '?'
    return str(val).strip().replace('.0', '') # Clean up floating point representation if any

def fmt_price(x):
    """Format harga: kalau NaN atau None -> '?'. Jika bilangan bulat besar -> no decimals, else 2 decimals."""
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return '?'
        v = float(x)
        if abs(v) >= 1:
            # tampilkan tanpa desimal jika sebenarnya bilangan bulat (atau besar)
            if abs(v - round(v)) < 0.0001:
                return f"{int(round(v)):,}"
            return f"{v:,.2f}"
        else:
            return f"{v:.4f}"
    except Exception:
        return '?'

def extract_trade_levels(r):
    """
    Return dict with buy_low, buy_high, buy_range_str, tp1,tp2,tp3, sl, rr.
    Will try multiple possible column names.
    """
    # buy area candidates
    buy_low = None
    buy_high = None
    buy_range_str = None
    for low_key, high_key in [('Buy_Area_Low','Buy_Area_High'),
                              ('Buy_Low','Buy_High'),
                              ('Buy_Area','Buy_Area'),
                              ('Fast_Buy_Range','Fast_Buy_Range'),
                              ('Buy_Range','Buy_Range')]:
        if low_key in r and high_key in r and pd.notna(r.get(low_key)) and pd.notna(r.get(high_key)) and low_key != high_key:
            buy_low = r.get(low_key)
            buy_high = r.get(high_key)
            break
    # fallback if single-string buy range present
    if (buy_low is None or buy_high is None) and ('Buy_Range' in r and pd.notna(r.get('Buy_Range'))):
        buy_range_str = str(r.get('Buy_Range'))
    if buy_range_str is None and buy_low is not None and buy_high is not None:
        buy_range_str = f"{fmt_price(buy_low)} - {fmt_price(buy_high)}"

    # TP candidates
    def first_present(*keys):
        for k in keys:
            if k in r and pd.notna(r.get(k)):
                return r.get(k)
        return None

    tp1 = first_present('TP1','T1','Target1','Fast_TP1','Fast_Target','Target')
    tp2 = first_present('TP2','T2','Target2','Fast_TP2')
    tp3 = first_present('TP3','T3','Target3','Fast_TP3')
    # if only single Target present, use as TP1
    if tp1 is None and ('Target' in r and pd.notna(r.get('Target'))):
        tp1 = r.get('Target')

    sl = first_present('SL','Stop','StopLoss','Fast_SL','SL_Price','stop')
    rr = first_present('RR_Ratio','Fast_RR_Ratio','RR_ratio','RR')

    return {
        'buy_low': buy_low,
        'buy_high': buy_high,
        'buy_range_str': buy_range_str or '?',
        'tp1': tp1,
        'tp2': tp2,
        'tp3': tp3,
        'sl': sl,
        'rr': rr
    }


def format_fast_row(r):
    trade = extract_trade_levels(r)
    rr_str = format_rr(trade.get('rr') or r.get('Fast_RR_Ratio', 0))
    buy_range = trade.get('buy_range_str') or get_str(r, 'Fast_Buy_Range')
    tp1 = fmt_price(trade.get('tp1'))
    tp2 = fmt_price(trade.get('tp2'))
    tp3 = fmt_price(trade.get('tp3'))
    sl_price = fmt_price(trade.get('sl') or r.get('Fast_SL'))

    return (
        f"*{r.get('ticker')}* (Score: {fmt_int(r.get('FastScore'))})\n"
        f"  - **Range Beli**: {buy_range} (Close {fmt_price(r.get('close'))})\n"
        f"  - **TP1/TP2/TP3**: {tp1} / {tp2} / {tp3}\n"
        f"  - **SL**: {sl_price} ({rr_str})"
    )

def format_swing_row(r):
    trade = extract_trade_levels(r)
    rr_str = format_rr(trade.get('rr') or r.get('Swing_RR_Ratio', 0))
    buy_range = trade.get('buy_range_str') or get_str(r, 'Swing_Buy_Range')
    tp1 = fmt_price(trade.get('tp1'))
    tp2 = fmt_price(trade.get('tp2'))
    tp3 = fmt_price(trade.get('tp3'))
    sl_price = fmt_price(trade.get('sl') or r.get('Swing_SL'))

    return (
        f"*{r.get('ticker')}* (Score: {fmt_int(r.get('SwingScore'))})\n"
        f"  - **Range Beli**: {buy_range} (Close {fmt_price(r.get('close'))})\n"
        f"  - **TP1/TP2/TP3**: {tp1} / {tp2} / {tp3}\n"
        f"  - **SL**: {sl_price} ({rr_str})"
    )

def format_kura_row(r):
    trade = extract_trade_levels(r)
    rr_str = format_rr(trade.get('rr') or r.get('RR_Ratio', 0))
    buy_range = trade.get('buy_range_str') or get_str(r, 'Buy_Range')
    tp1 = fmt_price(trade.get('tp1') or r.get('Target'))
    tp2 = fmt_price(trade.get('tp2'))
    tp3 = fmt_price(trade.get('tp3'))
    sl_price = fmt_price(trade.get('sl') or r.get('SL_Price'))

    return (
        f"*{r.get('ticker')}* (Close: {fmt_price(r.get('close'))})\n"
        f"  - **Range Beli**: {buy_range}\n"
        f"  - **TP1/TP2/TP3**: {tp1} / {tp2} / {tp3}\n"
        f"  - **SL**: {sl_price} ({rr_str})"
    )

def format_harmonic_row(row):
    t = row.get('Ticker') or row.get('ticker') or ''
    pat = row.get('Pattern') or row.get('pattern') or ''
    dirc = row.get('Direction') or row.get('direction') or ''
    # Allow multiple possible TP names in harmonic results
    t1 = row.get('Target1') or row.get('T1') or row.get('T_1') or row.get('TP1')
    t2 = row.get('Target2') or row.get('T2') or row.get('TP2')
    t3 = row.get('Target3') or row.get('T3') or None
    stop = row.get('Stop') or row.get('SL') or row.get('stop')

    t1_f = fmt_price(pd.to_numeric(t1, errors='coerce')) if t1 is not None else '?'
    t2_f = fmt_price(pd.to_numeric(t2, errors='coerce')) if t2 is not None else '?'
    t3_f = fmt_price(pd.to_numeric(t3, errors='coerce')) if t3 is not None else '?'
    stop_f = fmt_price(pd.to_numeric(stop, errors='coerce'))

    return (
        f"*{t or '?'}* | Pola: **{pat or '?'} ({dirc or '?'})**\n"
        f"  - TP1/TP2/TP3: {t1_f} / {t2_f} / {t3_f} | SL: {stop_f}"
    )


def format_alligator_row(r):
    # Asumsi data Alligator (dari FTScreenerV8) ada di GLOBAL_SCREENER_RESULTS
    # Karena Alligator tidak di-export, kita hanya bisa menggunakan data dari fast_top/swing_top jika kolomnya ada.
    # Kita pertahankan logic yang lama (menggunakan r.get)
    rr_str = format_rr(r.get('Alligator_RR_Ratio', 0))
    buy_range = get_str(r, 'Alligator_Buy_Range')
    tp_target = fmt_int(r.get('Alligator_Target'))
    sl_price = fmt_int(r.get('Alligator_SL'))

    return (
        f"*{r.get('ticker')}* | Sinyal: **{r.get('AlligatorSignal', '?')}** ({r.get('AlligatorStage', '?')})\n"
        f"  - **Range Beli**: {buy_range} (Close {fmt_int(r.get('close'))})\n"
        f"  - **Target (TP)**: {tp_target} | SL: {sl_price} ({rr_str})"
    )


def format_hns_row(r):
    # HnS/Inverse HnS uses target/stop from pattern geometry
    target = fmt_int(r.get('target'))
    stop = fmt_int(r.get('stop'))
    return (
        f"*{r.get('ticker')}* | Pola: **{r.get('type', '?')}**\n"
        f"  - Target: {target} | Stop: {stop}"
    )

def format_cnh_row(r):
    rr_str = format_rr(r.get('RR_Ratio', 0))
    buy_range = get_str(r, 'Buy_Range')
    target = fmt_int(r.get('target'))
    stop = fmt_int(r.get('stop')) # Uses 'stop' column

    return (
        f"*{r.get('ticker')}* | Pola: **{r.get('type', '?')}**\n"
        f"  - **Range Beli**: {buy_range} (Close {fmt_int(r.get('close'))})\n"
        f"  - **Target (TP)**: {target} | SL: {stop} ({rr_str})"
    )

def format_df(df, columns=None):
    """
    Format dataframe menjadi text Telegram (Markdown).
    df : pandas DataFrame
    columns : list kolom yang ingin ditampilkan berurutan
    """
    if df is None or df.empty:
        return "\n( kosong )\n"

    if columns is None:
        columns = list(df.columns)

    lines = []
    # Khusus untuk BSJP, format gap_up_prob sebagai persentase
    col_map = {'gap_up_prob': lambda x: f"{x:.2%}" if pd.notna(x) else '?'}

    for _, row in df.iterrows():
        parts = []
        # attempt to extract trade levels and show condensed trade info if present
        trade = extract_trade_levels(row)
        for col in columns:
            val = row.get(col)
            if pd.isna(val):
                val_str = "?"
            elif col in col_map:
                val_str = col_map[col](val)
            elif isinstance(val, (int, float)):
                if col in ['SorePagiScore']:
                    val_str = f"{val:.2f}"
                elif abs(val) > 1000:
                    val_str = f"{int(val):,}"
                else:
                    val_str = f"{val}"
            else:
                val_str = str(val)
            parts.append(f"*{col}*: {val_str}")

        # Add trade summary if buy area or TP/SL found
        trade_parts = []
        if trade.get('buy_range_str') and trade.get('buy_range_str') != '?':
            trade_parts.append(f"Buy: {trade.get('buy_range_str')}")
        # TP/SL
        tp1 = fmt_price(trade.get('tp1'))
        tp2 = fmt_price(trade.get('tp2'))
        tp3 = fmt_price(trade.get('tp3'))
        sl = fmt_price(trade.get('sl'))
        if tp1 != '?' or tp2 != '?' or tp3 != '?':
            trade_parts.append(f"TP: {tp1}/{tp2}/{tp3}")
        if sl != '?':
            trade_parts.append(f"SL: {sl}")

        if trade_parts:
            parts.append(" | ".join(trade_parts))

        lines.append(" | ".join(parts))

    return "\n".join(lines) + "\n"


# --- FUNGSI FORMATTING BARU KHUSUS JEPANG ---
def format_jp_row(df, columns):
    """
    Format dataframe Jepang menjadi text Telegram (Markdown) dengan kolom yang relevan.
    Using logic similar to format_df
    """
    if df is None or df.empty:
        return "\n( kosong )\n"

    lines = []
    # Convert dataframe rows to dictionary for easy access
    for _, row in df.iterrows():
        parts = []
        for col in columns:
            val = row.get(col)
            val_str = get_str(row, col)
            
            # Formatting khusus untuk harga dan nilai besar
            if col in ['close', 'volume', 'value']:
                 val_str = fmt_int(pd.to_numeric(val, errors='coerce'))
            
            parts.append(f"*{col}*: {val_str}")
        lines.append(" | ".join(parts))

    return "\n".join(lines) + "\n"


# --------------- HANDLER FUNCTIONS (FTScreener V8) -------------------

# --- HANDLER BARU UNTUK BSJP (NEW) ---
def handle_bsjp(chat_id):
    send_long_message(chat_id, "⏳ Memproses sinyal BSJP (Beli Sore Jual Pagi) dengan data real-time...")
    res = run_screener_if_needed()
    fast_df = res.get('fast_top', pd.DataFrame()) # Ambil data dasar dari Fast Screener
    
    if fast_df.empty:
        send_long_message(chat_id, "❌ Tidak ada hasil Fast Trading dasar. Gagal menjalankan BSJP.")
        return

    try:
        df_bsjp = run_bsjp_calculation(fast_df)
    except Exception as e:
        print(f"[BSJP] Error in calculation: {e}", traceback.format_exc())
        send_long_message(chat_id, f"❌ Gagal menghitung BSJP: {e}", parse_mode=None)
        return

    if df_bsjp.empty:
        send_long_message(chat_id, "❌ Tidak ada sinyal BSJP (Beli Sore Jual Pagi) yang memenuhi kriteria.")
        return
        
    df_top = df_bsjp.head(10).copy()
    message = "🌙 *BSJP Top 10 Picks (Real-Time)*\n"
    # Kolom: Ticker, Close EOD, Harga Realtime, Gap Up Prob, Score BSJP
    message += format_df(df_top, columns=['ticker', 'close', 'rt_price', 'gap_up_prob', 'SorePagiScore'])
    message += "\n\n*Detail Trade Levels (BSJP):*\n"
    for _, r in df_top.iterrows():
        # df_bsjp mungkin tidak punya Fast_* fields, gunakan generic format_kura_row or format_fast_row if available
        message += format_kura_row(r.to_dict()) + "\n"

    send_long_message(chat_id, message)
# --- END HANDLER BSJP ---


def handle_fast(chat_id):
    res = run_screener_if_needed()
    df = res.get('fast_top', pd.DataFrame()) # Gunakan .get untuk safety

    if df.empty:
        send_long_message(chat_id, "❌ Tidak ada sinyal Fast Trading terdeteksi.")
        return

    df_top = df.head(10).copy()
    message = "🚀 *Fast Trading Top 10 Scores*\n"
    # Jika ingin ringkasan kolom
    message += format_df(df_top, columns=['ticker', 'close', 'FastScore', 'volume', 'value'])
    # Tambah detail per-row (TP/SL/Buy) di bawah ringkasan
    message += "\n\n*Detail Trade Levels:*\n"
    for _, r in df_top.iterrows():
        message += format_fast_row(r.to_dict()) + "\n"
    send_long_message(chat_id, message)


def handle_swing(chat_id):
    res = run_screener_if_needed()
    df = res.get('swing_top', pd.DataFrame())

    if df.empty:
        send_long_message(chat_id, "❌ Tidak ada sinyal Swing Trading terdeteksi.")
        return

    df_top = df.head(10).copy()
    message = "⛵ *Swing Trading Top 10 Scores*\n"
    message += format_df(df_top, columns=['ticker', 'close', 'SwingScore', 'AccumScore', 'volume', 'value'])
    message += "\n\n*Detail Trade Levels:*\n"
    for _, r in df_top.iterrows():
        message += format_swing_row(r.to_dict()) + "\n"
    send_long_message(chat_id, message)


def handle_harmonic(chat_id):
    res = run_screener_if_needed()
    df = res.get('harmonic', pd.DataFrame())
    
    if df.empty:
        send_long_message(chat_id, "❌ Tidak ada Harmonic Patterns terdeteksi.")
        return
        
    message = "📊 *Harmonic Patterns Detected (Top RR Ratio)*\n"
    # Kolom: Ticker, Harga, Pattern, RR Ratio, Entry, TP1, SL
    message += format_df(df, columns=['ticker', 'close', 'pattern', 'RR_ratio', 'entry_high', 'TP1', 'SL'])
    send_long_message(chat_id, message)

def handle_kura(chat_id):
    res = run_screener_if_needed()
    df = res.get('kura', pd.DataFrame())
    
    if df.empty:
        send_long_message(chat_id, "❌ Tidak ada Kura-Kura Ninja terdeteksi.")
        return
        
    message = "🐢 *Kura-Kura Ninja Detected*\n"
    # Kolom: Ticker, Harga, KuraKuraNinja (Flag/Status), RankScore, FastScore, SwingScore
    message += format_df(df.head(10), columns=['ticker', 'close', 'KuraKuraNinja', 'RankScore', 'FastScore', 'SwingScore'])
    send_long_message(chat_id, message)

def handle_hns(chat_id):
    res = run_screener_if_needed()
    df = res.get('hns', pd.DataFrame())
    
    if df.empty:
        send_long_message(chat_id, "❌ Tidak ada Head and Shoulders (HnS) terdeteksi.")
        return
        
    message = "👤 *Head and Shoulders (HnS) Detected*\n"
    # Kolom: Ticker, Harga, HnS (Flag/Status), Target Harga, FastScore, SwingScore
    message += format_df(df.head(10), columns=['ticker', 'close', 'HnS', 'HnS_target', 'FastScore', 'SwingScore'])
    send_long_message(chat_id, message)

def handle_cnh(chat_id):
    res = run_screener_if_needed()
    df = res.get('cnh', pd.DataFrame())
    
    if df.empty:
        send_long_message(chat_id, "❌ Tidak ada Cup and Handle (CnH) terdeteksi.")
        return
        
    message = "☕ *Cup and Handle (CnH) Detected*\n"
    # Kolom: Ticker, Harga, CnH (Flag/Status), Target Harga, FastScore, SwingScore
    message += format_df(df.head(10), columns=['ticker', 'close', 'CnH', 'CnH_target', 'FastScore', 'SwingScore'])
    send_long_message(chat_id, message)

def handle_alligator(chat_id):
    res = run_screener_if_needed()
    # Asumsi data alligator ada di fast_top/swing_top atau key terpisah 'alligator'
    # Jika Alligator tidak tersedia, kita gunakan fast_top sebagai fallback
    df = res.get('alligator', res.get('fast_top', pd.DataFrame()))
    
    # Asumsi 'alligator' key berisi saham yang memenuhi kriteria Alligator (Uptrend)
    if df.empty:
        send_long_message(chat_id, "❌ Tidak ada sinyal Alligator (Uptrend) terdeteksi.")
        return
        
    df_top = df.head(10).copy()
    message = "🐊 *Alligator Trend (Top 10 Scores)*\n"
    # Kolom: Ticker, Harga, Status Trend (Alligator_Trend), FastScore, SwingScore
    message += format_df(df_top, columns=['ticker', 'close', 'Alligator_Trend', 'FastScore', 'SwingScore'])
    send_long_message(chat_id, message)


# --- HANDLER FUNCTIONS BARU KHUSUS JEPANG ---

def handle_ichimoku(chat_id):
    if not (jps_v3 or jps):
        send_long_message(chat_id, "❌ Modul JPScreener tidak tersedia.")
        return
    res = run_screener_if_needed()
    df = res.get('jp_ichimoku', res.get('ichimoku', pd.DataFrame())) 
    
    if df.empty:
        send_long_message(chat_id, "❌ Tidak ada sinyal Ichimoku Bullish (TK Crossover di atas Kumo) terdeteksi.")
        return
        
    df_top = df.head(10).copy()
    message = "☁️ *Ichimoku Bullish (TK Cross Over Kumo)*\n"
    message += format_jp_row(df_top, columns=['ticker', 'close', 'Ichimoku_Score', 'Ichimoku_Reasons', 'Buy_Area_Low', 'Buy_Area_High'])
    send_long_message(chat_id, message)

def handle_reversal(chat_id):
    if not (jps_v3 or jps):
        send_long_message(chat_id, "❌ Modul JPScreener tidak tersedia.")
        return
    res = run_screener_if_needed()
    df = res.get('jp_reversal', res.get('reversal', pd.DataFrame())) 
    
    if df.empty:
        send_long_message(chat_id, "❌ Tidak ada sinyal Candlestick Bullish Reversal terdeteksi.")
        return
        
    df_top = df.head(10).copy()
    message = "🕯️ *Candlestick Bullish Reversal*\n"
    # reversal module in v3 returns 'Reversals' or list of patterns; adapt display
    if 'Reversals' in df_top.columns:
        message += format_jp_row(df_top, columns=['ticker', 'close', 'Reversals'])
    else:
        # v3.2 uses 'Patterns' (list) in reversal_v3_2; we flatten if needed
        df_top_display = df_top.copy()
        if 'Patterns' in df_top_display.columns:
            df_top_display['Patterns'] = df_top_display['Patterns'].apply(lambda x: ';'.join([p.get('Pattern') for p in x]) if isinstance(x, list) else (x or ''))
            message += format_jp_row(df_top_display, columns=['ticker', 'close', 'Patterns'])
        else:
            message += format_jp_row(df_top, columns=['ticker', 'close'])
    send_long_message(chat_id, message)

def handle_heiken(chat_id):
    if not (jps_v3 or jps):
        send_long_message(chat_id, "❌ Modul JPScreener tidak tersedia.")
        return
    res = run_screener_if_needed()
    df = res.get('jp_heikenashi', res.get('heiken', pd.DataFrame())) 
    
    if df.empty:
        send_long_message(chat_id, "❌ Tidak ada sinyal Heiken Ashi Tren Kuat (Hijau tanpa sumbu bawah) terdeteksi.")
        return
        
    df_top = df.head(10).copy()
    message = " smoothing *Heiken Ashi Trend Kuat*\n"
    # heiken_v3 has HA_Color, HA_Consec, Buy_Area_Low/High
    cols = [c for c in ['ticker','close','HA_Color','HA_Consec','Buy_Area_Low','Buy_Area_High'] if c in df_top.columns]
    message += format_jp_row(df_top, columns=cols)
    send_long_message(chat_id, message)

def handle_continuation(chat_id):
    if not (jps_v3 or jps):
        send_long_message(chat_id, "❌ Modul JPScreener tidak tersedia.")
        return
    res = run_screener_if_needed()
    df = res.get('jp_continuation', res.get('continuation', pd.DataFrame()))
    if df.empty:
        send_long_message(chat_id, "❌ Tidak ada sinyal Continuation terdeteksi.")
        return
    # continuation entries may be nested dicts; format top-level info
    lines = []
    for idx, row in df.head(10).iterrows():
        t = row.get('ticker') or row.get('Ticker') or '?'
        kb = row.get('KijunBounce')
        bb = row.get('BB_HA_Continuation')
        parts = [f"*{t}*"]
        if isinstance(kb, dict):
            parts.append(f"KijunBounce: {kb.get('Buy_Area_Low')} - {kb.get('Buy_Area_High')}")
        if isinstance(bb, dict):
            parts.append(f"BB_HA: {bb.get('Buy_Area_Low')} - {bb.get('Buy_Area_High')}")
        lines.append(" | ".join(parts))
    send_long_message(chat_id, "🔁 *Continuation Signals (Top 10)*\n" + ("\n".join(lines)))

def handle_renko(chat_id):
    if not (jps_v3 or jps):
        send_long_message(chat_id, "❌ Modul JPScreener tidak tersedia.")
        return
    res = run_screener_if_needed()
    df = res.get('jp_renko', res.get('renko', pd.DataFrame()))
    if df.empty:
        send_long_message(chat_id, "❌ Tidak ada sinyal Renko (pseudo) terdeteksi.")
        return
    # Format: ticker, Renko_Signal, Buy_Area_Low/High, Brick, Move
    message = "⛏️ *Renko (pseudo) Signals*\n"
    cols = [c for c in ['ticker','Renko_Signal','Buy_Area_Low','Buy_Area_High','Brick','Move'] if c in df.columns]
    message += format_jp_row(df.head(10), columns=cols)
    send_long_message(chat_id, message)


def handle_jp_precision(chat_id):
    if not (jps_v3 or jps):
        send_long_message(chat_id, "❌ Modul JPScreener tidak tersedia.")
        return
    res = run_screener_if_needed()
    df = res.get('jp_combined', res.get('jp_precision', pd.DataFrame()))
    if df.empty:
        send_long_message(chat_id, "❌ Tidak ada sinyal Presisi Tinggi (Ichimoku + Heiken Ashi) terdeteksi.")
        return
    df_top = df.head(10).copy()
    message = "🎯 *Japan High Precision (Ichimoku + Heiken Ashi)*\n"
    message += format_jp_row(df_top, columns=['ticker', 'close', 'Composite_Score', 'Ichimoku_Reasons', 'HA_Buy_Low', 'HA_Buy_High'])
    send_long_message(chat_id, message)

# ---------------- JP new handlers: summary, full, refresh, ticker ----------------

def handle_jp_summary(chat_id):
    """Ringkasan singkat: top 10 combined dari JPScreenerV3.2"""
    res = run_screener_if_needed()
    df = res.get('jp_combined', pd.DataFrame())
    if df.empty:
        send_long_message(chat_id, "❌ Tidak ada hasil JP (combined) — pastikan JPScreenerV3html.py sudah ada dan data tersedia.")
        return
    top = df.head(10).copy()
    message = "🇯🇵 *JPScreenerV3.2 — Top Combined (Top 10)*\n"
    cols = [c for c in ['ticker','close','Composite_Score','Ichimoku_Buy_Low','Ichimoku_Buy_High','HA_Buy_Low','HA_Buy_High','Renko_Signal','Reversal_Patterns'] if c in top.columns]
    message += format_jp_row(top, columns=cols)
    send_long_message(chat_id, message)

def handle_jp_full(chat_id):
    """Kirim CSV combined_v3_2.csv ke chat (file) dan ringkasan top 20"""
    # Force run to ensure outputs exist (optional: we could use cached)
    res = run_screener_if_needed()
    df = res.get('jp_combined', pd.DataFrame())
    if df.empty:
        send_long_message(chat_id, "❌ Tidak ada hasil JP (combined) untuk dikirim.")
        return
    # send top summary
    top = df.head(20).copy()
    message = "🇯🇵 *JPScreenerV3.2 — Full Combined Top 20*\n"
    cols = [c for c in ['ticker','close','Composite_Score','Ichimoku_Buy_Low','Ichimoku_Buy_High','HA_Buy_Low','HA_Buy_High','Renko_Signal','Reversal_Patterns'] if c in top.columns]
    message += format_jp_row(top, columns=cols)
    send_long_message(chat_id, message)
    # attach CSV file if exists
    try:
        csv_path = Path.cwd() / 'outputs' / 'combined_v3_2.csv'
        if csv_path.exists():
            data_bytes = csv_path.read_bytes()
            send_document(chat_id, csv_path.name, data_bytes, caption="Combined results (JPScreenerV3.2)")
        else:
            send_long_message(chat_id, "⚠️ File outputs/combined_v3_2.csv tidak ditemukan — jalankan JPScreener dulu.")
    except Exception as e:
        send_long_message(chat_id, f"Failed to send CSV: {e}")

def handle_jp_refresh(chat_id):
    """Force refresh JP screener (re-run, heavier)."""
    try:
        send_long_message(chat_id, "🔄 Memulai refresh JPScreenerV3.2 — akan memakan beberapa detik...")
        res = run_screener_if_needed(force=True)
        send_long_message(chat_id, "✅ Refresh selesai. Gunakan /jp atau /jpfull untuk melihat hasil.")
    except Exception as e:
        send_long_message(chat_id, f"❌ Refresh gagal: {e}")

def handle_jp_ticker(chat_id, ticker):
    """Return single-row detail for given ticker from combined results."""
    if not ticker:
        send_long_message(chat_id, "Gunakan: /jp TICKER  (mis: /jp BBCA)")
        return
    res = run_screener_if_needed()
    df = res.get('jp_combined', pd.DataFrame())
    if df.empty:
        send_long_message(chat_id, "❌ Tidak ada hasil JP (combined). Jalankan /jprefresh jika perlu.")
        return
    # standardize ticker casing
    t = ticker.strip().upper()
    row = df[df['ticker'].astype(str).str.upper() == t]
    if row.empty:
        send_long_message(chat_id, f"❌ Ticker {t} tidak ditemukan di hasil combined.")
        return
    r = row.iloc[0].to_dict()
    # Build message
    lines = [f"🇯🇵 *Detail {t}*"]
    lines.append(f"Close: {fmt_int(r.get('close'))}")
    lines.append(f"Composite Score: {r.get('Composite_Score', '?')}")
    # Ichimoku buy area
    if r.get('Ichimoku_Buy_Low') is not None and r.get('Ichimoku_Buy_High') is not None:
        lines.append(f"Ichimoku Buy Area: {r.get('Ichimoku_Buy_Low')} - {r.get('Ichimoku_Buy_High')}")
    # HA
    if r.get('HA_Buy_Low') is not None and r.get('HA_Buy_High') is not None:
        lines.append(f"HA Buy Area: {r.get('HA_Buy_Low')} - {r.get('HA_Buy_High')}")
    # Renko
    if r.get('Renko_Buy_Low') is not None and r.get('Renko_Buy_High') is not None:
        lines.append(f"Renko Buy Area: {r.get('Renko_Buy_Low')} - {r.get('Renko_Buy_High')}")
    # Reversal patterns
    if r.get('Reversal_Patterns'):
        lines.append(f"Reversal Patterns: {r.get('Reversal_Patterns')}")
    # TP/SL
    lines.append(f"TP1 / TP2 / TP3: {fmt_int(r.get('TP1'))} / {fmt_int(r.get('TP2'))} / {fmt_int(r.get('TP3'))}")
    lines.append(f"SL: {fmt_int(r.get('SL'))}")
    # Bandar & Volume
    if 'Bandar_Condition' in r and r.get('Bandar_Condition'):
        lines.append(f"Bandar: {r.get('Bandar_Condition')} (Score {r.get('Bandar_Score')})")
    if 'Volume' in r:
        lines.append(f"Volume: {fmt_int(r.get('Volume'))} (Z: {r.get('Volume_Z')})")
    send_long_message(chat_id, "\n".join(lines))

# ---------------- Main polling ----------------------

def run_polling():
    print('[BOT] V16 polling loop started — interval', POLL_INTERVAL, 's')
    last_offset = load_offset()
    while True:
        try:
            updates = get_updates(offset=last_offset+1, timeout=30)
            for u in updates:
                uid = u.get('update_id')
                msg = u.get('message') or {}
                if not msg:
                    last_offset = max(last_offset, uid or last_offset)
                    continue
                chat = msg.get('chat', {})
                chat_id = chat.get('id')
                text = (msg.get('text') or '').strip()
                user = (msg.get('from') or {}).get('username')
                print(f"[TG] {chat_id} {user}: {text}")

                # only accept requests from allowed chat ids (tolerant int/str)
                if CHAT_IDS and not any(str(chat_id) == str(allowed) for allowed in CHAT_IDS):
                    print('[BOT] Ignoring message from', chat_id)
                    last_offset = max(last_offset, uid or last_offset)
                    continue

                if text.startswith('/'):
                    parts = text.split()
                    cmd = parts[0].lower()
                    args = parts[1:]
                    try:
                        if cmd == '/fast':
                            handle_fast(chat_id)
                        elif cmd == '/swing':
                            handle_swing(chat_id)
                        elif cmd == '/bsjp': # <<< HANDLER BSJP BARU DITAMBAHKAN
                            handle_bsjp(chat_id)
                        elif cmd == '/alligator':
                            handle_alligator(chat_id)
                        elif cmd in ['/kura','/kura-kura']:
                            handle_kura(chat_id)
                        elif cmd == '/hns':
                            handle_hns(chat_id)
                        elif cmd == '/cnh':
                            handle_cnh(chat_id)
                        elif cmd in ['/harmonic','/harmonics']:
                            handle_harmonic(chat_id)
                        # --- PERINTAH JEPANG BARU ---
                        elif cmd == '/ichimoku':
                            handle_ichimoku(chat_id)
                        elif cmd in ['/reversal']:
                            handle_reversal(chat_id)
                        elif cmd in ['/heiken','/heikenashi']:
                            handle_heiken(chat_id)
                        elif cmd in ['/continuation']:
                            handle_continuation(chat_id)
                        elif cmd in ['/renko']:
                            handle_renko(chat_id)
                        elif cmd in ['/jpprecision']:
                            handle_jp_precision(chat_id)
                        elif cmd == '/jp':
                            # /jp or /jp TICKER
                            if args:
                                handle_jp_ticker(chat_id, args[0])
                            else:
                                handle_jp_summary(chat_id)
                        elif cmd == '/jpfull':
                            handle_jp_full(chat_id)
                        elif cmd == '/jprefresh':
                            handle_jp_refresh(chat_id)
                        # --- END PERINTAH JEPANG BARU ---
                        elif cmd in ['/help','/start']:
                            # <<< PESAN /HELP DIPERBARUI >>>
                            send_long_message(chat_id, "👋 Halo! Saya MIDAS siap bantu screening saham.\nGunakan:\n/fast /swing /bsjp /alligator /kura /hns /cnh /harmonic /ichimoku /reversal /heiken /continuation /renko /jpprecision /jp /jpfull /jprefresh /help")
                        else:
                            send_long_message(chat_id, 'Perintah tidak dikenal. Ketik /help untuk daftar perintah.')
                    except Exception as e:
                        print('[HANDLER] exception', e, traceback.format_exc())
                        # Kirim error tanpa format Markdown untuk menghindari Bad Request
                        send_long_message(chat_id, f'Error executing {cmd}: {e}', parse_mode=None)

                last_offset = max(last_offset, uid or last_offset)
            save_offset(last_offset)
            time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            print('[BOT] Stopped by user')
            break
        except Exception as e:
            print('[BOT] Polling error', e, traceback.format_exc())
            time.sleep(5)

if __name__ == '__main__':
    run_polling()