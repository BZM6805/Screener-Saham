# ===============================================
# FAST & SWING TRADING v5 — ADVANCED LOGIC (v9)
# +++ FULL ONLINE VERSION (YFINANCE BASED) +++
# +++ NO LOCAL DATA REQUIRED +++
# ===============================================
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from functools import lru_cache
import yfinance as yf 
from dataclasses import dataclass
from datetime import datetime, timedelta

# ----------------- Config -----------------
st.set_page_config(page_title="FAST & SWING FULL ONLINE", layout="wide")

# Harmonic params
PIVOT_WINDOW = 3
MIN_DIST = 2
TOL = 0.06
MAX_D_CANDLE_DISTANCE = 5

# Default Tickers (LQ45 / Populer) agar user tidak perlu ketik satu2 di awal
DEFAULT_TICKERS = """
BBCA, BBRI, BMRI, BBNI, TLKM, ASII, UNTR, ICBP, INDF, UNVR, 
GOTO, ARTO, BUKA, EMTK, MDKA, ADRO, PTBA, ITMG, PGAS, INCO, 
ANTM, TINS, BRIS, BBTN, CTRA, SMRA, BSDE, PWON, JSMR, KLBF, 
CPIN, JPFA, SIDO, MAPI, ACES, ERAA, AMRT, MEDC, AKRA, HRUM,
INKP, TKIM, SMGR, INTP, EXCL, ISAT, TLKM
"""

@dataclass
class PatternPoint:
    index: int
    price: float

# ================= Helper Data Online ==================
def get_online_data(ticker_list, period="1y"):
    """
    Mengambil data historis dari Yahoo Finance
    """
    # Bersihkan ticker dan tambahkan .JK jika belum ada
    clean_tickers = []
    for t in ticker_list:
        t = t.upper().strip()
        if not t: continue
        if not t.endswith(".JK"):
            t += ".JK"
        clean_tickers.append(t)
    
    if not clean_tickers:
        return pd.DataFrame()

    try:
        # Download data bulk
        print(f"Downloading {len(clean_tickers)} tickers...")
        data = yf.download(clean_tickers, period=period, group_by='ticker', auto_adjust=True, threads=True)
        
        stack_data = []
        
        # Jika hanya 1 ticker, strukturnya beda (tidak multi-index di level 0)
        if len(clean_tickers) == 1:
            df_single = data.copy()
            df_single['ticker'] = clean_tickers[0].replace(".JK", "")
            df_single = df_single.reset_index()
            stack_data.append(df_single)
        else:
            # Multi-ticker
            for ticker_jk in clean_tickers:
                try:
                    # Akses data per ticker dari MultiIndex column
                    df_t = data[ticker_jk].copy()
                    # Cek apakah dataframe kosong/semua NaN
                    if df_t.empty or df_t['Close'].isna().all():
                        continue
                        
                    df_t['ticker'] = ticker_jk.replace(".JK", "")
                    df_t = df_t.reset_index()
                    stack_data.append(df_t)
                except KeyError:
                    continue
        
        if not stack_data:
            return pd.DataFrame()

        # Gabungkan semua
        all_df = pd.concat(stack_data, axis=0)
        
        # Standarisasi nama kolom agar sesuai logika lama
        # YFinance columns: Date, Open, High, Low, Close, Volume
        all_df.columns = [c.lower() for c in all_df.columns]
        all_df.rename(columns={'date': 'date', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}, inplace=True)
        
        # Tambahkan kolom dummy untuk compatibilitas script lama (karena YF gratis tidak ada data bandar)
        all_df['value'] = all_df['close'] * all_df['volume'] # Estimasi Value
        all_df['foreign_buy'] = 0
        all_df['foreign_sell'] = 0
        all_df['foreign'] = 0
        
        return all_df

    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return pd.DataFrame()

# ================= ENRICH & INDICATORS ==================
def enrich(df):
    df = df.sort_values('date').copy()
    # Pastikan numeric
    for col in ['open','high','low','close','volume','value']:
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

    # Accumulation (Modifikasi untuk Online: Value based, bukan Foreign Flow)
    df['accum_1d'] = df['value'].fillna(0)
    df['accum_5d'] = df['value'].rolling(5, min_periods=1).mean()
    df['accum_20d'] = df['value'].rolling(20, min_periods=1).mean()

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

# ================= NON-HARMONIC PATTERN DETECTION =================
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


# ================= HARMONIC PATTERN (V9) =================
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

    for i in range(len(all_pivots) - 4):
        X, A, B, C, D = all_pivots[i:i+5]
        if (len(df) - 1 - D.index) > MAX_D_CANDLE_DISTANCE:
            continue
        if X.price <= A.price: continue 

        ratio_AB_XA = ratio(A.price - B.price, X.price - A.price)
        ratio_BC_AB = ratio(B.price - C.price, A.price - B.price)
        ratio_AD_XA = ratio(A.price - D.price, X.price - A.price)
        
        # Gartley
        if (X.price > A.price and C.price < A.price):
            if is_approx(ratio_AB_XA, 0.618) and (0.382 <= ratio_BC_AB <= 0.886) and is_approx(ratio_AD_XA, 0.786):
                patterns.append({'type': 'Bullish Gartley', 'D_price': D.price, 'date': df.iloc[D.index]['date']})
        # Butterfly
        if (X.price > A.price and C.price < A.price):
            if is_approx(ratio_AB_XA, 0.786) and (0.382 <= ratio_BC_AB <= 0.886) and is_approx(ratio_AD_XA, 1.27):
                patterns.append({'type': 'Bullish Butterfly', 'D_price': D.price, 'date': df.iloc[D.index]['date']})
        # Crab
        if (X.price > A.price and C.price < A.price):
            if (ratio_AB_XA <= 0.618) and (0.382 <= ratio_BC_AB <= 0.886) and is_approx(ratio_AD_XA, 1.618):
                patterns.append({'type': 'Bullish Crab', 'D_price': D.price, 'date': df.iloc[D.index]['date']})
        # Bat
        if (X.price > A.price and C.price < A.price):
            if (0.382 <= ratio_AB_XA <= 0.5) and (0.382 <= ratio_BC_AB <= 0.886) and is_approx(ratio_AD_XA, 0.886):
                patterns.append({'type': 'Bullish Bat', 'D_price': D.price, 'date': df.iloc[D.index]['date']})
                
    return patterns

# ================= KURA-KURA NINJA =================
def kura_kura_ninja(df):
    df = df.sort_values("date").copy()
    if len(df) < 21:
        return {"KuraKuraNinja": False}
    df['Highest20'] = df['high'].rolling(20, min_periods=1).max()
    df['Lowest20'] = df['low'].rolling(20, min_periods=1).min()
    df['Breakout'] = df['close'] > df['Highest20'].shift(1)
    df['Pullback'] = (df['close'] < df['Highest20']) & (df['close'] > df['Lowest20'])
    df['VolMA20'] = df['volume'].rolling(20, min_periods=1).mean()
    df['VolOK'] = df['volume'] > df['VolMA20']
    df['MoneyFlow'] = df['close'] * df['volume']
    df['MFMA20'] = df['MoneyFlow'].rolling(20, min_periods=1).mean()
    df['FlowOK'] = df['MoneyFlow'] > df['MFMA20']
    df['KuraKuraNinja'] = (df['Breakout'] & df['Pullback'] & df['VolOK'] & df['FlowOK']).fillna(False)
    return {"KuraKuraNinja": bool(df.iloc[-1]['KuraKuraNinja'])}

# ================= SCORING =================
def compute_score_fast(df):
    df = df.copy()
    # Flags
    df['Flag_SMA_support'] = ((df['close'] >= df['SMA50']) & (df['close'] <= df['SMA50']*1.03) & (df['SMA50'] > df['SMA200'])).astype(int)
    df['Flag_MACross'] = ((df['SMA5'] > df['SMA20']) & (df['SMA20'] > df['SMA50'])).astype(int)
    df['Flag_Breakout'] = ((df['close'] > df['high'].rolling(20, min_periods=1).max().shift(1)) & (df['volume'] > 1.5*df['vol_sma20'])).astype(int)
    df['Flag_VolSurge'] = (df['volume'] > 2*df['vol_sma20']).astype(int)
    # Accumulation based on Value only for Online version
    df['Flag_Accum'] = ((df['value'] > df['val_sma20'])).astype(int)

    df['FastScore'] = (
        df['Flag_SMA_support']*10 + df['Flag_MACross']*12 + df['Flag_Breakout']*15 + df['Flag_VolSurge']*8 + df['Flag_Accum']*20
    )
    df['FastScore'] = df['FastScore'].fillna(0).astype(int)
    df['SetupFast'] = df.apply(lambda r: ', '.join([k for k,v in {
        'SMA Support': r['Flag_SMA_support'], 'MA Cross': r['Flag_MACross'], 'Breakout': r['Flag_Breakout'], 'Vol Surge': r['Flag_VolSurge'], 'Accum': r['Flag_Accum']}.items() if v]), axis=1)
    return df

def compute_score_swing(row, patterns_for_ticker=None):
    score = 0
    logic = []
    if row.get('close',0) > row.get('EMA8',0): score += 10; logic.append('Close>EMA8')
    if row.get('EMA13',0) > row.get('SMA20',0): score += 10; logic.append('EMA13>SMA20')
    if row.get('SMA50',0) > row.get('SMA200',0): score += 10; logic.append('SMA50>SMA200')
    if row.get('volume',0) > 2*row.get('vol_sma20',1): score += 10; logic.append('Vol Surge')
    if row.get('val_sma20',0) > 1_000_000_000: score += 5; logic.append('Likuiditas OK')
    if row.get('StochK',50) > row.get('StochD',50) and row.get('StochK',50) < 80: score += 10; logic.append('Stoch Bullish')

    if patterns_for_ticker:
        for p in patterns_for_ticker:
            if 'Bullish' in p: score += 15; logic.append(f'Bullish: {p}')
                
    score = max(0, min(100, score))
    return {'SwingScore': int(score), 'SwingLogic': ', '.join(logic)}

# ================= UI & MAIN PIPELINE =================
st.title("📊 Full Online Stock Screener (No Upload)")
st.sidebar.header("Pengaturan")

input_tickers = st.sidebar.text_area("Daftar Ticker (pisahkan dengan koma/spasi)", DEFAULT_TICKERS, height=200)

if st.sidebar.button("🚀 Mulai Screening"):
    
    # Parse Tickers
    ticker_list = [t.strip() for t in input_tickers.replace(',', ' ').split() if t.strip()]
    
    with st.spinner(f"Downloading data for {len(ticker_list)} stocks..."):
        all_df = get_online_data(ticker_list)
        
    if all_df.empty:
        st.error("Gagal mengambil data. Cek koneksi internet atau nama ticker.")
        st.stop()

    st.success(f"Berhasil download data: {len(all_df)} baris.")

    # Processing
    with st.spinner('Memproses data teknikal...'):
        enriched = all_df.groupby('ticker', group_keys=False).apply(enrich).reset_index(drop=True)
        enriched = enriched.groupby('ticker', group_keys=False).apply(compute_alligator).reset_index(drop=True)

    # Patterns
    patterns_map = {}
    hns_list_detail = []
    cnh_list_detail = []
    harmonic_bullish_list = []
    
    progress_bar = st.progress(0)
    unique_tickers = enriched['ticker'].unique()
    
    for i, t in enumerate(unique_tickers):
        df_t = enriched[enriched['ticker']==t].sort_values('date')
        if df_t.empty: continue
        
        pats = []
        
        # Detect
        hs = detect_head_and_shoulders(df_t, inverse=False)
        inv = detect_head_and_shoulders(df_t, inverse=True)
        cup = detect_cup_and_handle(df_t)
        harm = detect_harmonic_patterns(df_t)
        
        if hs.get('found'): 
            pats.append(hs['type'])
            hns_list_detail.append({'ticker':t, 'HnS':hs['type'], 'HnS_target':hs.get('target'), 'close':df_t.iloc[-1]['close']})
        if inv.get('found'): 
            pats.append(inv['type'])
            hns_list_detail.append({'ticker':t, 'HnS':inv['type'], 'HnS_target':inv.get('target'), 'close':df_t.iloc[-1]['close']})
        if cup.get('found'): 
            pats.append(cup['type'])
            cnh_list_detail.append({'ticker':t, 'CnH':cup['type'], 'CnH_target':cup.get('target'), 'close':df_t.iloc[-1]['close']})
            
        for h in harm:
            if 'Bullish' in h['type']:
                pats.append(h['type'])
                h_data = df_t.iloc[-1].to_dict()
                h_data['ticker'] = t
                h_data['HarmonicPattern'] = h['type']
                h_data['PatternDate'] = h['date'].strftime('%Y-%m-%d')
                h_data['PatternClosePrice'] = h['D_price']
                harmonic_bullish_list.append(h_data)
        
        patterns_map[t] = pats
        progress_bar.progress((i + 1) / len(unique_tickers))

    # Kura-Kura
    ninja_list = []
    for t in unique_tickers:
        df_t = enriched[enriched['ticker']==t].sort_values('date')
        res = kura_kura_ninja(df_t)
        res['ticker'] = t
        ninja_list.append(res)
    
    df_ninja = pd.DataFrame(ninja_list)
    enriched = enriched.merge(df_ninja, on='ticker', how='left')

    # Finalizing Latest Data
    latest_date = enriched['date'].max()
    df_latest = enriched[enriched['date']==latest_date].copy()
    
    # Scores
    df_latest = compute_score_fast(df_latest)
    swing_results = df_latest.apply(lambda r: compute_score_swing(r, patterns_for_ticker=patterns_map.get(r['ticker'], [])), axis=1)
    SwingDF = pd.DataFrame(list(swing_results))
    df_latest = pd.concat([df_latest.reset_index(drop=True), SwingDF.reset_index(drop=True)], axis=1)

    # Display
    st.markdown(f"### 📅 Data Terakhir: {latest_date.strftime('%Y-%m-%d')}")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["⚡ Fast Trade","🎯 Swing Trade","⭐ Harmonic","🐊 Alligator","🧑‍🦱 HnS","☕ Cup&Handle","🐢 Kura-Kura"])
    
    with tab1:
        st.subheader("Fast Trade Screener")
        cols = ['ticker','close','FastScore','SetupFast','volume','value']
        st.dataframe(df_latest.sort_values('FastScore', ascending=False)[cols], use_container_width=True)
        
    with tab2:
        st.subheader("Swing Trade Screener")
        cols = ['ticker','close','SwingScore','SwingLogic','volume']
        st.dataframe(df_latest.sort_values('SwingScore', ascending=False)[cols], use_container_width=True)

    with tab3:
        st.subheader("Harmonic Patterns")
        if harmonic_bullish_list:
            st.dataframe(pd.DataFrame(harmonic_bullish_list)[['ticker','HarmonicPattern','PatternClosePrice','PatternDate']], use_container_width=True)
        else:
            st.info("Tidak ada pola Harmonic ditemukan.")

    with tab4:
        st.subheader("Alligator")
        st.dataframe(df_latest[['ticker','close','AlligatorStage','AlligatorSignal']], use_container_width=True)

    with tab5:
        st.subheader("Head & Shoulders")
        if hns_list_detail: st.dataframe(pd.DataFrame(hns_list_detail), use_container_width=True)
        else: st.info("Tidak ada HnS ditemukan.")

    with tab6:
        st.subheader("Cup & Handle")
        if cnh_list_detail: st.dataframe(pd.DataFrame(cnh_list_detail), use_container_width=True)
        else: st.info("Tidak ada Cup & Handle ditemukan.")
        
    with tab7:
        st.subheader("Kura-Kura Ninja")
        kura = df_latest[df_latest['KuraKuraNinja']==True]
        if not kura.empty: st.dataframe(kura[['ticker','close','volume','val_sma20']], use_container_width=True)
        else: st.info("Tidak ada setup Kura-Kura Ninja.")
        
else:
    st.info("👈 Klik 'Mulai Screening' di sidebar untuk memulai.")