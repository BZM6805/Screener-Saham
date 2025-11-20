# JPScreenerV2.py
import pandas as pd
import numpy as np
import ta.trend as trend
from pathlib import Path

# --- PASTIKAN FTSCREENERV8 BERADA DI FOLDER YANG SAMA ---
try:
    import FTScreenerV8 as fts
except ImportError:
    print("ERROR: FTScreenerV8.py tidak ditemukan. Pastikan file berada di folder yang sama.")
    fts = None

# --- JAPANESE INDICATOR CALCULATION ---

def calculate_heiken_ashi(df):
    df = df.copy()
    # HA Close: Rata-rata candle saat ini
    df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # HA Open: Inisialisasi list untuk performa lebih cepat daripada iterasi .loc
    ha_open = [0.0] * len(df)
    
    # Baris pertama HA Open biasanya diambil dari Open/Close candle pertama
    ha_open[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    
    # Loop untuk menghitung HA Open selanjutnya
    # HA Open baris ini = (HA Open prev + HA Close prev) / 2
    ha_close_values = df['HA_Close'].values
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i-1] + ha_close_values[i-1]) / 2
        
    df['HA_Open'] = ha_open
    df['HA_High'] = df[['high', 'HA_Open', 'HA_Close']].max(axis=1)
    df['HA_Low'] = df[['low', 'HA_Open', 'HA_Close']].min(axis=1)
    return df

def calc_levels(close):
    tp1 = int(close * 1.02)
    tp2 = int(close * 1.04)
    tp3 = int(close * 1.06)
    sl  = int(close * 0.97)
    return tp1, tp2, tp3, sl

def get_ichimoku_signal(df):
    if df.empty or len(df) < 52:
        return None
    try:
        ichimoku = trend.IchimokuIndicator(
            high=df['high'], low=df['low'],
            window1=9, window2=26, window3=52
        )
        df['tenkan'] = ichimoku.ichimoku_conversion_line()
        df['kijun'] = ichimoku.ichimoku_base_line()
        df['senkou_a'] = ichimoku.ichimoku_a()
        df['senkou_b'] = ichimoku.ichimoku_b()
        df['Kumo_Max'] = df[['senkou_a','senkou_b']].max(axis=1)
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Signal: Tenkan cross above Kijun AND Price above Cloud
        tk_cross = (last['tenkan'] > last['kijun']) and (prev['tenkan'] <= prev['kijun'])
        above_kumo = last['close'] > last['Kumo_Max']
        
        if tk_cross and above_kumo:
            tp1,tp2,tp3,sl = calc_levels(last['close'])
            return {
                'ticker': df['ticker'].iloc[-1],
                'close': last['close'],
                'Signal_Type': 'Ichimoku TK Cross',
                'Score': 95,
                'Kumo_Status': 'Above Cloud',
                'Buy_Range': f"{int(last['kijun'])}-{int(last['tenkan'])}",
                'TP1': tp1, 'TP2': tp2, 'TP3': tp3, 'SL': sl
            }
    except Exception:
        return None
    return None

def get_heiken_ashi_signal(df):
    try:
        df_ha = calculate_heiken_ashi(df)
        last = df_ha.iloc[-1]
        
        # Signal: Green Candle (Close > Open) and No Lower Wick (Low == Open)
        if last['HA_Close'] > last['HA_Open'] and last['HA_Low'] == last['HA_Open']:
            tp1,tp2,tp3,sl = calc_levels(df['close'].iloc[-1])
            return {
                'ticker': df['ticker'].iloc[-1],
                'close': df['close'].iloc[-1],
                'Signal_Type': 'HA Strong Buy',
                'Score': 90,
                'Desc': 'Strong_Green_Continuation',
                'Buy_Range': f"{int(last['HA_Open'])}-{int(last['HA_Close'])}",
                'TP1': tp1, 'TP2': tp2, 'TP3': tp3, 'SL': sl
            }
    except Exception:
        return None
    return None

#=== Additional signals: Renko, Reversal (candlestick), Continuation (MA + HA) ===

def get_reversal_signal(df):
    """
    Detect simple bullish candlestick reversal patterns:
    - Bullish Engulfing
    - Hammer
    """
    if df is None or len(df) < 3:
        return None
    try:
        d = df.copy().tail(3).reset_index(drop=True)
        for c in ['open','high','low','close']:
            d[c] = pd.to_numeric(d[c], errors='coerce')
            
        prev = d.loc[1]
        last = d.loc[2]
        
        # 1. Bullish Engulfing
        prev_red = prev['close'] < prev['open']
        last_green = last['close'] > last['open']
        engulfs = (last['close'] >= prev['open']) and (last['open'] <= prev['close'])
        
        if prev_red and last_green and engulfs:
            tp1,tp2,tp3,sl = calc_levels(last['close'])
            return {
                'ticker': df['ticker'].iloc[-1],
                'close': last['close'],
                'Signal_Type': 'Bullish Engulfing',
                'Score': 85,
                'Buy_Range': f"{int(last['open'])}-{int(last['close'])}",
                'TP1': tp1, 'TP2': tp2, 'TP3': tp3, 'SL': sl
            }
            
        # 2. Hammer: body small, lower wick >= 2x body
        body = abs(last['close'] - last['open'])
        lower_wick = last['open'] - last['low'] if last['open'] > last['close'] else last['close'] - last['low']
        upper_wick = last['high'] - max(last['close'], last['open'])
        
        # Avoid Doji (body almost 0) unless significant wick
        if body > 0 and lower_wick >= 2 * body and upper_wick <= body*0.5:
            tp1,tp2,tp3,sl = calc_levels(last['close'])
            return {
                'ticker': df['ticker'].iloc[-1],
                'close': last['close'],
                'Signal_Type': 'Hammer',
                'Score': 75,
                'Buy_Range': f"{int(last['low'])}-{int(last['close'])}",
                'TP1': tp1, 'TP2': tp2, 'TP3': tp3, 'SL': sl
            }
    except Exception:
        return None
    return None

def get_continuation_signal(df):
    """Detect simple continuation: short MA above long MA and last 2 Heiken-Ashi green"""
    if df is None or len(df) < 30:
        return None
    try:
        d = df.copy().sort_values('date').tail(60).reset_index(drop=True)
        d['close'] = pd.to_numeric(d['close'], errors='coerce')
        d['ema5'] = d['close'].ewm(span=5, adjust=False).mean()
        d['ema20'] = d['close'].ewm(span=20, adjust=False).mean()
        
        if pd.isna(d['ema5'].iloc[-1]) or pd.isna(d['ema20'].iloc[-1]):
            return None
            
        ema_cross = d['ema5'].iloc[-1] > d['ema20'].iloc[-1]
        
        # Heiken Ashi check
        d_ha = calculate_heiken_ashi(d)
        last = d_ha.iloc[-1]
        prev = d_ha.iloc[-2]
        
        ha_green = (last['HA_Close'] > last['HA_Open']) and (prev['HA_Close'] > prev['HA_Open'])
        # No lower wick OR very small lower wick
        ha_no_lower = (last['HA_Low'] == last['HA_Open']) or ((last['HA_Close'] - last['HA_Low']) < (last['HA_Close']-last['HA_Open'])*0.3)
        
        if ema_cross and ha_green and ha_no_lower:
            tp1,tp2,tp3,sl = calc_levels(d['close'].iloc[-1])
            return {
                'ticker': df['ticker'].iloc[-1],
                'close': d['close'].iloc[-1],
                'Signal_Type': 'MA+HA Continuation',
                'Score': 88,
                'Desc': 'Continuation',
                'Buy_Range': f"{int(last['HA_Open'])}-{int(last['HA_Close'])}",
                'TP1': tp1, 'TP2': tp2, 'TP3': tp3, 'SL': sl
            }
    except Exception:
        return None
    return None

def get_renko_signal(df):
    """
    Simple Renko-like breakout detection using ATR-based brick size:
    If price moves by >= 2 bricks from previous close -> signal
    """
    if df is None or len(df) < 20:
        return None
    try:
        d = df.copy().sort_values('date').reset_index(drop=True)
        high = pd.to_numeric(d['high'], errors='coerce')
        low = pd.to_numeric(d['low'], errors='coerce')
        close = pd.to_numeric(d['close'], errors='coerce')
        
        # ATR Calculation
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=1).mean().iloc[-1]
        
        if pd.isna(atr) or atr <= 0:
            return None
            
        brick = atr
        prev_close = close.iloc[-2]
        last_close = close.iloc[-1]
        move = last_close - prev_close
        
        # Bullish renko if move >= 2 bricks
        if move >= 2 * brick:
            tp1,tp2,tp3,sl = calc_levels(last_close)
            return {
                'ticker': df['ticker'].iloc[-1],
                'close': last_close,
                'Signal_Type': 'Renko Breakout',
                'Score': 80,
                'Renko_Move': float(move),
                'Brick': float(brick),
                'Buy_Range': f"{int(prev_close)}-{int(last_close)}",
                'TP1': tp1, 'TP2': tp2, 'TP3': tp3, 'SL': sl
            }
    except Exception:
        return None
    return None

def run_all_jp_screeners():
    if not fts:
        return {}
    
    print("Mengambil data dari FTScreener...")
    df_all = fts.load_all_data()
    
    # List untuk menampung hasil
    ichimoku_list = []
    heiken_list = []
    reversal_list = []
    continuation_list = []
    renko_list = []
    
    print("Memproses indikator Japanese Technicals...")
    for ticker, df_t in df_all.groupby('ticker'):
        try:
            # Ambil data 200 bar terakhir
            df_t = df_t.sort_values('date').tail(200)
            if len(df_t) < 20: continue

            # 1. Ichimoku
            ichi = get_ichimoku_signal(df_t)
            if ichi: ichimoku_list.append(ichi)
            
            # 2. Heiken Ashi
            ha = get_heiken_ashi_signal(df_t)
            if ha: heiken_list.append(ha)
            
            # 3. Reversal (Candlestick)
            rev = get_reversal_signal(df_t)
            if rev: reversal_list.append(rev)
            
            # 4. Continuation
            cont = get_continuation_signal(df_t)
            if cont: continuation_list.append(cont)
            
            # 5. Renko
            ren = get_renko_signal(df_t)
            if ren: renko_list.append(ren)
            
        except Exception as e:
            # Skip ticker yang error datanya
            continue

    # Convert lists to DataFrames
    df_ich = pd.DataFrame(ichimoku_list)
    df_hei = pd.DataFrame(heiken_list)
    df_rev = pd.DataFrame(reversal_list)
    df_cont = pd.DataFrame(continuation_list)
    df_ren = pd.DataFrame(renko_list)
    
    # Gabungan Precision (Contoh: Irisan Ichimoku + Heiken)
    df_prec = pd.DataFrame()
    if not df_ich.empty and not df_hei.empty:
        # Ganti nama kolom agar tidak bentrok saat merge
        ich_sub = df_ich[['ticker','close','Score','Buy_Range','TP1']].rename(columns={'Score':'Ichi_Score'})
        hei_sub = df_hei[['ticker','Score']].rename(columns={'Score':'HA_Score'})
        
        df_prec = pd.merge(ich_sub, hei_sub, on='ticker', how='inner')
        if not df_prec.empty:
            df_prec['Total_Score'] = df_prec['Ichi_Score'] + df_prec['HA_Score']
            df_prec = df_prec.sort_values('Total_Score', ascending=False)

    return {
        'ichimoku': df_ich,
        'heiken': df_hei,
        'reversal': df_rev,
        'continuation': df_cont,
        'renko': df_ren,
        'jp_precision': df_prec
    }

if __name__ == '__main__':
    print("--- Running JPScreenerV2 Fix ---")
    results = run_all_jp_screeners()
    
    for key, df in results.items():
        print(f"\n=== {key.upper()} (Top 5) ===")
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(df.head(5).to_string(index=False))
        else:
            print("No signals found.")