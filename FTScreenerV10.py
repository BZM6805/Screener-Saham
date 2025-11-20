# ===============================================
# FAST & SWING TRADING v5 — ADVANCED LOGIC (v9)
# +++ INTEGRATED BSJP + Harmonic Bullish + V9 Swing/Kura Logic +++
# +++ FINAL FIXED (ModuleNotFoundError, KeyError, NameError Solved) +++
# ===============================================
import os, io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
from functools import lru_cache
import yfinance as yf 
from dataclasses import dataclass
from datetime import datetime

# ----------------- Config -----------------
# Gunakan path default atau sesuaikan jika perlu
DATA_DIR = Path(r"C:\Users\ADVAN\Downloads\eod_uploader\EODUploader\Bandar Detector\data_harian")
st.set_page_config(page_title="FAST & SWING TRADING v5 + ADVANCED LOGIC", layout="wide")

# Harmonic params dari V9
PIVOT_WINDOW = 3
MIN_DIST = 2
TOL = 0.06
MAX_D_CANDLE_DISTANCE = 5

@dataclass
class PatternPoint:
    index: int
    price: float

# ================= Normalisasi Kolom ==================
COL_MAP = {
    "Kode Saham": "ticker", "Kode": "ticker", "Ticker": "ticker", "Symbol": "ticker",
    "Nama Perusahaan": "company_name",
    "PrevClose": "prev_close", "Harga Sebelumnya": "prev_close",
    "Open Price": "open", "Open": "open",
    "Tertinggi": "high", "High": "high",
    "Terendah": "low", "Low": "low",
    "Penutupan": "close", "Close": "close", "Harga": "close",
    "Volume": "volume",
    "Nilai": "value", "Value": "value",
    "Frekuensi": "freq",
    "Foreign Buy": "foreign_buy", "FB": "foreign_buy",
    "Foreign Sell": "foreign_sell", "FS": "foreign_sell",
    "Net Foreign": "foreign",
    "BandarFlag": "bandarflag",
    "AccumHistory": "accum_hist",
    "Top_Broker": "top_broker", "TopBroker": "top_broker",
    "TopBrokerHist": "top_broker_hist",
    "Tanggal Perdagangan Terakhir": "date", "Tanggal": "date", "TradingDate": "date"
}

def normalize_columns(df):
    df = df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})
    if "date" in df.columns:
        bulan_map = {"Jan":"01","Feb":"04","Mar":"03","Apr":"04","Mei":"05","Jun":"06","Jul":"07","Agt":"08","Sep":"09","Okt":"10","Nov":"11","Des":"12"}
        def parse_date_id(x):
            if pd.isna(x): return pd.NaT
            x = str(x).strip()
            for k, v in bulan_map.items():
                if k in x:
                    x = x.replace(k, v)
            x = x.replace(" ", "-")
            return pd.to_datetime(x, errors="coerce", dayfirst=True)
        df["date"] = df["date"].apply(parse_date_id)
    return df

def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

# ================= ENRICH & INDICATORS ==================
def enrich(df):
    df = df.sort_values('date').copy()
    # basic numeric conversions
    for col in ['open','high','low','close','volume','value','foreign_buy','foreign_sell']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['prev_close'] = df['close'].shift(1)

    # Moving averages
    df['SMA5'] = df['close'].rolling(5, min_periods=1).mean()
    df['SMA20'] = df['close'].rolling(20, min_periods=1).mean()
    df['SMA50'] = df['close'].rolling(50, min_periods=1).mean()
    df['SMA200'] = df['close'].rolling(200, min_periods=1).mean()
    df['EMA8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['EMA13'] = df['close'].ewm(span=13, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['EMA60'] = df['close'].ewm(span=60, adjust=False).mean()

    # volume/value helpers
    df['vol_sma5'] = df['volume'].rolling(5, min_periods=1).mean()
    df['vol_sma20'] = df['volume'].rolling(20, min_periods=1).mean()
    df['val_sma20'] = df['value'].rolling(20, min_periods=1).mean()

    # ATR14
    high_low = df['high'] - df['low']
    high_prev = (df['high'] - df['prev_close']).abs()
    low_prev = (df['low'] - df['prev_close']).abs()
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14, min_periods=1).mean().ffill().fillna(0)

    # Stochastic
    low_min = df['low'].rolling(14, min_periods=1).min()
    high_max = df['high'].rolling(14, min_periods=1).max()
    denom = (high_max - low_min).replace(0, np.nan)
    df['StochK'] = 100 * (df['close'] - low_min) / denom
    df['StochK'] = df['StochK'].ffill().fillna(50)
    df['StochD'] = df['StochK'].rolling(3, min_periods=1).mean()

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema12 - ema26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd_line'] - df['macd_signal']

    # accumulation/value based rolling
    if 'foreign' not in df.columns:
        if 'foreign_buy' in df.columns and 'foreign_sell' in df.columns:
            df['foreign'] = df['foreign_buy'].fillna(0) - df['foreign_sell'].fillna(0)
        else:
            df['foreign'] = 0
    df['accum_1d'] = df['value'].fillna(0)
    df['accum_5d'] = df['value'].rolling(5, min_periods=1).mean()
    df['accum_20d'] = df['value'].rolling(20, min_periods=1).mean()

    # simple pattern placeholders
    df['HnS'] = np.nan
    df['CnH'] = np.nan

    return df

# Alligator helpers
def smma(series, period):
    return series.ewm(alpha=1/period, adjust=False).mean()

def compute_alligator(df):
    df = df.sort_values('date').copy()
    df['Lips']  = smma(df['close'],5).shift(3)
    df['Teeth'] = smma(df['close'],8).shift(5)
    df['Jaw']   = smma(df['close'],13).shift(8)

    def stage(row):
        lips, teeth, jaw = row['Lips'], row['Teeth'], row['Jaw']
        if pd.isna(lips) or pd.isna(teeth) or pd.isna(jaw):
            return None
        if abs(lips - teeth) < 0.2 and abs(teeth - jaw) < 0.2:
            return "Sleeping"
        if lips > teeth > jaw:
            return "Eating Uptrend"
        if jaw > teeth > lips:
            return "Eating Downtrend"
        if (lips > teeth and lips > jaw):
            return "Waking Up"
        if (lips < teeth and lips < jaw):
            return "Waking Down"
        return "Sated"

    df['AlligatorStage'] = df.apply(stage, axis=1)
    df['AlligatorSignal'] = None
    for i in range(1, len(df)):
        prev = df.iloc[i-1]['AlligatorStage']
        curr = df.iloc[i]['AlligatorStage']
        if curr == "Waking Up" and prev in ["Sleeping", "Sated", None]:
            df.loc[df.index[i], 'AlligatorSignal'] = "BUY"
        elif curr in ["Waking Down","Sated"] and prev in ["Eating Uptrend","Waking Up"]:
            df.loc[df.index[i], 'AlligatorSignal'] = "SELL"
    return df

# ================= NON-HARMONIC PATTERN DETECTION (dari V5) =================
def find_local_maxima(series, order=3):
    return [i for i in range(order, len(series)-order) if series[i] == max(series[i-order:i+order+1])]

def find_local_minima(series, order=3):
    return [i for i in range(order, len(series)-order) if series[i] == min(series[i-order:i+order+1])]

def detect_head_and_shoulders(data, lookback=80, inverse=False):
    highs = data['high'].values[-lookback:]
    lows  = data['low'].values[-lookback:]
    if len(highs) < 10:
        return {"found": False}
    peaks = find_local_maxima(highs)
    troughs = find_local_minima(lows)
    if not inverse:
        if len(peaks) >= 3 and len(troughs) >= 2:
            L,H,R = peaks[-3:]
            t1,t2 = troughs[-2:]
            left, head, right = highs[L], highs[H], highs[R]
            if head > left and head > right and abs(left-right)/head < 0.05:
                neckline = (lows[t1] + lows[t2]) / 2
                target = neckline - (head - neckline)
                stoploss = right*1.01
                return {"found":True,"type":"H&S Bearish","neckline":neckline,"target":target,"stoploss":stoploss}
    else:
        if len(troughs) >= 3 and len(peaks) >= 2:
            L,H,R = troughs[-3:]
            p1,p2 = peaks[-2:]
            left, head, right = lows[L], lows[H], lows[R]
            if head < left and head < right and abs(left-right)/abs(head) < 0.05:
                neckline = (highs[p1] + highs[p2]) / 2
                target = neckline + (neckline - head)
                stoploss = right*0.99
                return {"found": True, "type": "Inverse H&S Bullish", "neckline": neckline, "target": target, "stoploss": stoploss}
    return {"found":False}

def detect_cup_and_handle(data, lookback=120):
    closes = data['close'].values[-lookback:]
    highs  = data['high'].values[-lookback:]
    lows   = data['low'].values[-lookback:]
    vols   = data['volume'].values[-lookback:]
    if len(closes) < 30:
        return {"found":False}
    resistance = np.max(closes)
    support = np.min(closes)
    cup_height = resistance - support
    cond_depth = (cup_height / max(resistance, 1e-9)) > 0.12
    cond_shape = closes[:len(closes)//2].mean() > support and closes[-len(closes)//2:].mean() > support
    handle_lows = np.min(closes[-10:])
    cond_handle = (resistance - handle_lows) < (cup_height * 0.33)
    breakout_vol = vols[-1] > max(vols[-20:].mean(), 1) if len(vols) >= 20 else True
    if cond_depth and cond_shape and cond_handle and breakout_vol:
        return {"found":True,"type":"Cup & Handle Bullish","breakout":resistance,"target":resistance+cup_height,"stoploss":handle_lows}
    return {"found":False}


# ================= HARMONIC PATTERN (dari V9) =================
# Fungsi bantuan untuk menghitung rasio Fibonacci
def ratio(a, b):
    if b == 0: return np.inf
    return abs(a - b) / abs(b)

def is_approx(val, target, tolerance=TOL):
    return abs(val - target) / target <= tolerance

def find_pivots_harmonic(series, window=PIVOT_WINDOW):
    pivots = []
    for i in range(window, len(series) - window):
        is_high = series[i] == max(series[i - window : i + window + 1])
        is_low = series[i] == min(series[i - window : i + window + 1])
        if is_high or is_low:
            pivots.append(i)
    return [PatternPoint(i, series[i]) for i in pivots]

def detect_harmonic_patterns(df, n_pivots=5):
    patterns = []
    if len(df) < 50: return patterns

    highs_pivots = find_pivots_harmonic(df['high'].values, window=PIVOT_WINDOW)
    lows_pivots = find_pivots_harmonic(df['low'].values, window=PIVOT_WINDOW)
    
    all_pivots = sorted(highs_pivots + lows_pivots, key=lambda p: p.index)
    all_pivots = [p for i, p in enumerate(all_pivots) if i==0 or p.index - all_pivots[i-1].index >= MIN_DIST] 

    if len(all_pivots) < 5: return patterns

    # Analisis 5-titik (X, A, B, C, D) untuk Bullish Patterns
    for i in range(len(all_pivots) - 4):
        X, A, B, C, D = all_pivots[i:i+5]
        
        if (len(df) - 1 - D.index) > MAX_D_CANDLE_DISTANCE:
            continue
            
        # Bullish pattern: X > A (turun)
        if X.price <= A.price:
             continue 

        # Hitung rasio untuk 4 tipe Bullish Pattern
        ratio_AB_XA = ratio(A.price - B.price, X.price - A.price)
        ratio_BC_AB = ratio(B.price - C.price, A.price - B.price)
        ratio_AD_XA = ratio(A.price - D.price, X.price - A.price)
        
        # Bullish Gartley (B: 0.618 XA, D: 0.786 XA)
        if (X.price > A.price and C.price < A.price):
            is_B = is_approx(ratio_AB_XA, 0.618)
            is_C = (ratio_BC_AB >= 0.382 and ratio_BC_AB <= 0.886)
            is_D = is_approx(ratio_AD_XA, 0.786)
            
            if is_B and is_C and is_D:
                patterns.append({'type': 'Bullish Gartley', 'D_price': D.price, 'date': df.iloc[D.index]['date']})
                
        # Bullish Butterfly (B: 0.786 XA, D: 1.27 XA)
        if (X.price > A.price and C.price < A.price):
            is_B = is_approx(ratio_AB_XA, 0.786)
            is_C = (ratio_BC_AB >= 0.382 and ratio_BC_AB <= 0.886)
            is_D = is_approx(ratio_AD_XA, 1.27)
            
            if is_B and is_C and is_D:
                patterns.append({'type': 'Bullish Butterfly', 'D_price': D.price, 'date': df.iloc[D.index]['date']})
        
        # Bullish Crab (B: <= 0.618 XA, D: 1.618 XA)
        if (X.price > A.price and C.price < A.price):
            is_B = (ratio_AB_XA <= 0.618)
            is_C = (ratio_BC_AB >= 0.382 and ratio_BC_AB <= 0.886)
            is_D = is_approx(ratio_AD_XA, 1.618)
            
            if is_B and is_C and is_D:
                patterns.append({'type': 'Bullish Crab', 'D_price': D.price, 'date': df.iloc[D.index]['date']})
                
        # Bullish Bat (B: 0.382-0.5 XA, D: 0.886 XA)
        if (X.price > A.price and C.price < A.price):
            is_B = (ratio_AB_XA >= 0.382 and ratio_AB_XA <= 0.5)
            is_C = (ratio_BC_AB >= 0.382 and ratio_BC_AB <= 0.886)
            is_D = is_approx(ratio_AD_XA, 0.886)
            
            if is_B and is_C and is_D:
                patterns.append({'type': 'Bullish Bat', 'D_price': D.price, 'date': df.iloc[D.index]['date']})
                
    return patterns


# ================= KURA-KURA NINJA (V9 LOGIC) =================
def kura_kura_ninja(df):
    df = df.sort_values("date").copy()
    if len(df) < 21:
        return {"KuraKuraNinja": False}
    df['Highest20'] = df['high'].rolling(20, min_periods=1).max()
    df['Lowest20'] = df['low'].rolling(20, min_periods=1).min()
    df['Breakout'] = df['close'] > df['Highest20'].shift(1)  # FIXED
    df['Pullback'] = (df['close'] < df['Highest20']) & (df['close'] > df['Lowest20'])
    df['VolMA20'] = df['volume'].rolling(20, min_periods=1).mean()
    df['VolOK'] = df['volume'] > df['VolMA20']
    df['MoneyFlow'] = df['close'] * df['volume']
    df['MFMA20'] = df['MoneyFlow'].rolling(20, min_periods=1).mean()
    df['FlowOK'] = df['MoneyFlow'] > df['MFMA20']
    df['KuraKuraNinja'] = (df['Breakout'] & df['Pullback'] & df['VolOK'] & df['FlowOK']).fillna(False)
    return {"KuraKuraNinja": bool(df.iloc[-1]['KuraKuraNinja'])}

# ================= SCORING: FAST & SWING =================
def compute_score_fast(df):
    # Logika Fast Score V5 dipertahankan
    df = df.copy()
    df['value'] = df['close'] * df['volume']
    df['vol_sma20'] = df['volume'].rolling(20, min_periods=1).mean().fillna(0)
    df['val_sma20'] = df['value'].rolling(20, min_periods=1).mean().fillna(0)

    # Flags
    df['Flag_SMA_support'] = ((df['close'] >= df['SMA50']) & (df['close'] <= df['SMA50']*1.03) & (df['SMA50'] > df['SMA200'])).astype(int)
    df['Flag_MACross'] = ((df['SMA5'] > df['SMA20']) & (df['SMA20'] > df['SMA50'])).astype(int)
    df['Flag_Breakout'] = ((df['close'] > df['high'].rolling(20, min_periods=1).max().shift(1)) & (df['volume'] > 1.5*df['vol_sma20'])).astype(int)
    df['Flag_VolSurge'] = (df['volume'] > 2*df['vol_sma20']).astype(int)
    df['Flag_Accum'] = ((df['accum_1d'] > 0) & (df['accum_5d'] > df['accum_20d'])).astype(int)

    df['FastScore'] = (
        df['Flag_SMA_support']*10 + df['Flag_MACross']*12 + df['Flag_Breakout']*15 + df['Flag_VolSurge']*8 + df['Flag_Accum']*20
    )
    df['FastScore'] = df['FastScore'].fillna(0).astype(int)
    df['SetupFast'] = df.apply(lambda r: ', '.join([k for k,v in {
        'SMA Support': r['Flag_SMA_support'], 'MA Cross': r['Flag_MACross'], 'Breakout': r['Flag_Breakout'], 'Vol Surge': r['Flag_VolSurge'], 'Accum': r['Flag_Accum']}.items() if v]), axis=1)
    return df

def compute_score_swing(row, patterns_for_ticker=None):
    # === LOGIKA SWING SCORE V9 ===
    score = 0
    logic = []
    
    # 1. Posisi Harga terhadap MA
    if row.get('close',0) > row.get('EMA8',0):
        score += 10; logic.append('Close>EMA8')
    if row.get('EMA13',0) > row.get('SMA20',0):
        score += 10; logic.append('EMA13>SMA20')
    if row.get('SMA50',0) > row.get('SMA200',0):
        score += 10; logic.append('SMA50>SMA200 (Golden Cross)')

    # 2. Volume & Likuiditas
    if row.get('volume',0) > 2*row.get('vol_sma20',1):
        score += 10; logic.append('Vol Surge (2x)')
    if row.get('val_sma20',0) > 1_000_000_000: # Likuiditas > 1M
        score += 5; logic.append('Likuiditas OK')

    # 3. Akumulasi
    if row.get('accum_5d',0) > row.get('accum_20d',0):
        score += 15; logic.append('Accum 5D > 20D')

    # 4. Stochastic
    if row.get('StochK',50) > row.get('StochD',50) and row.get('StochK',50) < 80:
        score += 10; logic.append('Stoch Bullish')

    # 5. Pola Candlestick/Chart
    if patterns_for_ticker:
        for p in patterns_for_ticker:
            if 'Bullish' in p:
                score += 15; logic.append(f'Bullish Pattern: {p}')
            elif 'Bearish' in p:
                score -= 10; logic.append(f'Bearish Pattern: {p}')
                
    score = max(0, min(100, score))
    return {'SwingScore': int(score), 'SwingLogic': ', '.join(logic), 'SetupSwing': ', '.join(logic)}

# ================= BSJP Functions (dari V9) =================
def fetch_realtime(ticker_list):
    data = {}
    # yfinance requires .JK suffix for IDX stocks
    full_ticker_list = [t + ".JK" for t in ticker_list]
    # Set timeout for yfinance download
    yf_data = yf.download(full_ticker_list, period="2d", interval="1d", progress=False, timeout=10)
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
            last = info.iloc[-1]
            data[t] = {
                'rt_price': float(last['Close']),
                'rt_volume': float(last['Volume'])
            }
    return data

def rekomendasi_sore_pagi(latest_df, top_n=5):
    tickers = latest_df['ticker'].unique().tolist()
    liquid_tickers = latest_df[latest_df['val_sma20'] > 1e6]['ticker'].tolist() 
    top_tickers_fast = latest_df.sort_values('FastScore', ascending=False).head(50)['ticker'].tolist()
    tickers_to_fetch = list(set(liquid_tickers) & set(top_tickers_fast))

    try:
        rt = fetch_realtime(tickers_to_fetch)
    except Exception as e:
        st.warning(f"Gagal mengambil data real-time (yfinance): {e}")
        return pd.DataFrame()
        
    df = latest_df[latest_df['ticker'].isin(rt.keys())].copy().reset_index(drop=True)
    df['rt_price'] = df['ticker'].apply(lambda x: rt.get(x,{}).get('rt_price', np.nan))
    df = df.dropna(subset=['rt_price', 'close'])
    
    df['gap_up_prob'] = ((df['rt_price'] - df['close']) / df['close']).fillna(0)
    df['evening_strength'] = (
        (df['EMA13'] > df['SMA20']).astype(int) +
        (df['volume'] > df['vol_sma20']).astype(int) +
        (df['close'] > df['EMA8']).astype(int)
    )
    df['SorePagiScore'] = (df['gap_up_prob'] * 100 * 0.4) + (df['evening_strength'] * 0.4) + (df['FastScore'] / 100 * 0.2)
    
    df = df.sort_values('SorePagiScore', ascending=False).head(top_n)
    return df[['ticker','close','rt_price','gap_up_prob','evening_strength','FastScore','SorePagiScore']]

# ================= PIPELINE =================
folder = st.sidebar.text_input("Path folder (.xlsx files)", value=str(DATA_DIR))
if not folder or not os.path.isdir(folder):
    st.error('Folder data tidak ditemukan atau tidak valid.')
    st.stop()

files = sorted([p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in ('.xlsx','.xls')])
if not files:
    st.error('Tidak ada file data EOD valid ditemukan.')
    st.stop()

DFs = []
for f in files:
    try:
        dft = pd.read_excel(f)
        dft = normalize_columns(dft)
        DFs.append(dft)
    except Exception:
        continue

if not DFs:
    st.error('No valid data files')
    st.stop()

all_df = pd.concat(DFs, ignore_index=True)
all_df = ensure_cols(all_df, ['ticker','open','high','low','close','volume','value','date'])
all_df['date'] = pd.to_datetime(all_df['date'], errors='coerce')
all_df = all_df.dropna(subset=['ticker','date','close'])
all_df['ticker'] = all_df['ticker'].astype(str).str.strip()
all_df = all_df.sort_values(['ticker','date'])

# Proses Enrich, Alligator, dan Patterns
with st.spinner('Memproses data teknikal...'):
    enriched = all_df.groupby('ticker', group_keys=False).apply(enrich).reset_index(drop=True)
    enriched = enriched.groupby('ticker', group_keys=False).apply(compute_alligator).reset_index(drop=True)

@lru_cache(maxsize=None)
def cached_patterns(tkr, df_json):
    df_t = pd.read_json(df_json, orient='records')
    pats = []
    
    # Non-Harmonic Patterns 
    hs = detect_head_and_shoulders(df_t, lookback=80, inverse=False)
    inv = detect_head_and_shoulders(df_t, lookback=80, inverse=True)
    cup = detect_cup_and_handle(df_t, lookback=120)
    for p in [hs, inv, cup]:
        if p.get('found'):
            pats.append(p['type'])
            
    # Harmonic Patterns
    harmonic_res = detect_harmonic_patterns(df_t, n_pivots=5)
    for h in harmonic_res:
        if 'Bullish' in h['type']:
             pats.append(h['type'])
             
    return pats, harmonic_res 

patterns_map = {}
hns_list_detail = []
cnh_list_detail = []
harmonic_bullish_list = []

# Iterasi untuk mendapatkan semua pattern detail
for t in enriched['ticker'].unique():
    df_t = enriched[enriched['ticker']==t].sort_values('date')
    if df_t.empty: continue
    
    j = df_t.to_json(orient='records')
    pats, harmonic_res = cached_patterns(t, j)
    patterns_map[t] = pats
    
    # Capture HnS/CnH details
    hs = detect_head_and_shoulders(df_t, lookback=80, inverse=False)
    inv = detect_head_and_shoulders(df_t, lookback=80, inverse=True)
    cup = detect_cup_and_handle(df_t, lookback=120)

    for h in [hs, inv]:
        if h.get('found'):
            hns_list_detail.append({'ticker': t, 'HnS': h['type'], 'HnS_target': h.get('target', np.nan), 'close': df_t.iloc[-1]['close']})
    
    if cup.get('found'):
        cnh_list_detail.append({'ticker': t, 'CnH': 'Cup & Handle Bullish', 'CnH_target': cup.get('target', np.nan), 'close': df_t.iloc[-1]['close']})

    # Capture Harmonic Bullish details
    for h in harmonic_res:
        if 'Bullish' in h['type']:
            h_data = df_t.iloc[-1].copy() 
            h_data['ticker'] = t
            h_data['HarmonicPattern'] = h['type']
            h_data['PatternDate'] = h['date'].strftime('%Y-%m-%d')
            h_data['PatternClosePrice'] = h['D_price']
            harmonic_bullish_list.append(h_data.to_dict())


ninja_list = []
for t in enriched['ticker'].unique():
    df_t = enriched[enriched['ticker']==t].sort_values('date')
    res = kura_kura_ninja(df_t)
    res['ticker'] = t
    ninja_list.append(res)

df_ninja = pd.DataFrame(ninja_list)
if 'KuraKuraNinja' not in df_ninja.columns:
    df_ninja['KuraKuraNinja'] = False

enriched = enriched.merge(df_ninja, on='ticker', how='left')

latest_date = enriched['date'].max()
st.markdown(f"### 📅 Data terakhir: {latest_date.strftime('%Y-%m-%d')}")
st.title("📊 Stock Screener — Fast & Swing v5 + ADVANCED LOGIC")

df_latest = enriched[enriched['date']==latest_date].copy()
if df_latest.empty:
    st.error('No data for latest date')
    st.stop()

# Menghitung Fast Score
df_latest = compute_score_fast(df_latest)

# Menghitung Swing Score (V9 Logic)
swing_results = df_latest.apply(lambda r: compute_score_swing(r, patterns_for_ticker=patterns_map.get(r['ticker'], [])), axis=1)
SwingDF = pd.DataFrame(list(swing_results))
df_latest = pd.concat([df_latest.reset_index(drop=True), SwingDF.reset_index(drop=True)], axis=1)

df_latest['AccumScore'] = (
    (df_latest['accum_1d']>0).astype(int) + (df_latest['accum_5d']>0).astype(int) + (df_latest['accum_20d']>0).astype(int)
) * 5

# ================= TABS & DISPLAY =================
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["🌙 BSJP","⭐ Harmonic Bullish","⚡ Fast Trade","🎯 Swing Trade","📊 Chart","🐊 Alligator","🧑‍🦱 HnS","☕ Cup&Handle","🐢 Kura-Kura"])

# TAB 0 - BSJP
with tab0:
    st.subheader('🌙 Beli Sore — ☀️ Jual Pagi Recommendations (BSJP)')
    try:
        df_latest_for_bsjp = df_latest[df_latest['FastScore'] > 0].copy()
        if not df_latest_for_bsjp.empty:
            with st.spinner('Mengambil data real-time via yfinance...'):
                rec = rekomendasi_sore_pagi(df_latest_for_bsjp, top_n=10)
            if not rec.empty:
                st.dataframe(rec, use_container_width=True)
                st.download_button('⬇️ Download BSJP (CSV)', rec.to_csv(index=False).encode('utf-8'), 'bsjp_v5_v9.csv', 'text/csv')
            else:
                st.info('Tidak ada rekomendasi BSJP yang dihasilkan atau gagal mengambil data real-time.')
        else:
            st.info('Tidak ada saham dengan FastScore > 0 untuk dianalisis BSJP.')
    except Exception as e:
        st.error(f"Gagal menghasilkan rekomendasi BSJP: {e}")

# TAB 1 - HARMONIC BULLISH
with tab1:
    st.subheader('⭐ Bullish Harmonic Pattern')
    df_harmonic = pd.DataFrame(harmonic_bullish_list)
    if not df_harmonic.empty:
        # Merge dengan skor terbaru
        df_harmonic = df_harmonic.drop_duplicates(subset=['ticker','HarmonicPattern']).merge(
            df_latest[['ticker', 'close', 'FastScore', 'SwingScore', 'AccumScore']], 
            on='ticker', 
            suffixes=('_pattern', '_latest'),
            how='inner'
        )
        df_harmonic = df_harmonic.sort_values(['HarmonicPattern', 'SwingScore'], ascending=[True, False])
        
        # DEFINE COLS HARMONGIC DISPLAY
        cols = ['ticker', 'HarmonicPattern', 'PatternDate', 'PatternClosePrice', 'close_latest', 'SwingScore', 'FastScore', 'AccumScore']

        # Rename columns for cleaner display
        df_display = df_harmonic[cols].rename(columns={'close_latest': 'LatestClose', 'PatternClosePrice': 'D_PointPrice'})
        st.dataframe(df_display, use_container_width=True)
        st.download_button('⬇️ Download Harmonic Bullish (CSV)', df_display.to_csv(index=False).encode('utf-8'), 'harmonic_bullish_v9.csv', 'text/csv')
    else:
        st.info('❌ Tidak ada pola Harmonic Bullish terdeteksi.')

# TAB 2 - FAST
with tab2:
    st.subheader('⚡ Fast Trade Screener')
    cols = ['ticker','close','FastScore','SetupFast','EMA8','EMA21','SMA50','volume','value']
    st.dataframe(df_latest.sort_values('FastScore', ascending=False)[cols].reset_index(drop=True))
    st.download_button('⬇️ Download Fast (CSV)', df_latest[cols].to_csv(index=False).encode('utf-8'), 'fast_v5_v9.csv', 'text/csv')

# TAB 3 - SWING
with tab3:
    st.subheader('🎯 Swing Screener (V9 Logic)')
    cols = ['ticker','close','SwingScore','SwingLogic','EMA13','EMA21','EMA60','volume','value']
    st.dataframe(df_latest.sort_values('SwingScore', ascending=False)[cols].reset_index(drop=True))
    st.download_button('⬇️ Download Swing (CSV)', df_latest[cols].to_csv(index=False).encode('utf-8'), 'swing_v5_v9.csv', 'text/csv')

# TAB 4 - CHART
with tab4:
    st.subheader('📊 Chart Visualisasi')
    tickers = st.multiselect('Pilih ticker', options=df_latest['ticker'].unique().tolist(), default=df_latest.sort_values('SwingScore', ascending=False)['ticker'].head(3).tolist())
    for t in tickers:
        df_t = enriched[enriched['ticker']==t].sort_values('date').tail(200)
        if df_t.empty: continue
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df_t['date'], df_t['close'], label='Close')
        ax.plot(df_t['date'], df_t['SMA20'], linestyle='--', label='SMA20')
        ax.plot(df_t['date'], df_t['SMA50'], label='SMA50')
        ax.set_title(t)
        ax.legend()
        st.pyplot(fig)

# TAB 5 - ALLIGATOR
with tab5:
    st.subheader('🐊 Alligator Screener')
    allig = df_latest.copy()
    allig['TradeAdvice'] = allig.apply(lambda r: ('✅ Layak Beli' if r['AlligatorSignal']=='BUY' and r['value']>5_000_000_000 else ('⚠️ Uptrend tapi likuiditas kurang' if r['AlligatorSignal']=='BUY' else ('❌ Hindari' if r['AlligatorSignal']=='SELL' else '🔎 Netral'))), axis=1)
    sort_key = allig['AlligatorSignal'].apply(lambda x: 0 if x=='BUY' else (1 if x=='SELL' else 2))
    allig = allig.assign(SortKey=sort_key).sort_values(['SortKey','value'], ascending=[True,False])
    st.dataframe(allig[['ticker','close','AlligatorStage','AlligatorSignal','volume','value','TradeAdvice','AccumScore']])
    st.download_button('⬇️ Download Alligator (CSV)', allig.to_csv(index=False).encode('utf-8'), 'alligator_v5_v9.csv', 'text/csv')

# TAB 6 - HEAD & SHOULDERS
with tab6:
    st.subheader('🧑‍🦱 Head & Shoulders')
    hns_df = pd.DataFrame(hns_list_detail)
    if not hns_df.empty:
        hns_df = hns_df.merge(df_latest[['ticker', 'FastScore', 'SwingScore', 'AccumScore']], on='ticker', how='left')
        st.dataframe(hns_df[['ticker','close','HnS','HnS_target','FastScore','SwingScore','AccumScore']])
        st.download_button('⬇️ Download HnS (CSV)', hns_df.to_csv(index=False).encode('utf-8'), 'hns_v5_v9.csv', 'text/csv')
    else:
        st.info('❌ Tidak ada pola HnS terdeteksi')

# TAB 7 - CUP & HANDLE
with tab7:
    st.subheader('☕ Cup & Handle')
    cnh_df = pd.DataFrame(cnh_list_detail)
    if not cnh_df.empty:
        cnh_df = cnh_df.merge(df_latest[['ticker', 'FastScore', 'SwingScore', 'AccumScore']], on='ticker', how='left')
        st.dataframe(cnh_df[['ticker','close','CnH','CnH_target','FastScore','SwingScore','AccumScore']])
        st.download_button('⬇️ Download CnH (CSV)', cnh_df.to_csv(index=False).encode('utf-8'), 'cnh_v5_v9.csv', 'text/csv')
    else:
        st.info('❌ Tidak ada Cup&Handle terdeteksi')

# TAB 8 - KURA-KURA NINJA
with tab8:
    st.subheader('🐢 Kura-Kura Ninja Screener (V9 Logic)')
    df_kura = df_latest[df_latest['KuraKuraNinja']==True].copy()
    if not df_kura.empty:
        df_kura['RankScore'] = (df_kura['value'].rank(pct=True)*40 + df_kura['FastScore'].rank(pct=True)*30 + df_kura['AccumScore'].rank(pct=True)*30).round(2)
        df_kura = df_kura.sort_values('RankScore', ascending=False)
        st.dataframe(df_kura[['ticker','close','volume','value','val_sma20','FastScore','SwingScore','AccumScore','RankScore']])
        st.download_button('⬇️ Download Kura-Kura (CSV)', df_kura.to_csv(index=False).encode('utf-8'), 'kura_kura_v5_v9.csv', 'text/csv')
    else:
        st.info('❌ Tidak ada saham memenuhi Kura-Kura Ninja')