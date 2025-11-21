import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# --- KONFIGURASI ---
# Daftar saham default (Bisa anda tambah/kurang)
DEFAULT_TICKERS = [
    "BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "UNTR", "ICBP", "INDF", "UNVR",
    "GOTO", "ARTO", "BUKA", "EMTK", "MDKA", "ADRO", "PTBA", "ITMG", "PGAS", "INCO",
    "ANTM", "TINS", "BRIS", "BBTN", "CTRA", "SMRA", "BSDE", "PWON", "JSMR", "KLBF",
    "CPIN", "JPFA", "SIDO", "MAPI", "ACES", "ERAA", "AMRT", "MEDC", "AKRA", "HRUM"
]

def get_data(tickers):
    print(f"⏳ Sedang mengambil data untuk {len(tickers)} saham...")
    clean_tickers = [t + ".JK" for t in tickers if not t.endswith(".JK")]
    
    try:
        # Download data hari ini + data intraday terakhir
        df = yf.download(clean_tickers, period="5d", group_by='ticker', threads=True, progress=False)
        return df
    except Exception as e:
        print(f"Error download: {e}")
        return None

def analyze_bsjp(df_all, tickers):
    results = []
    print("⚙️  Menganalisis sinyal BSJP...")
    
    for t in tickers:
        try:
            t_jk = t + ".JK"
            if t_jk not in df_all.columns.levels[0]: continue
            
            data = df_all[t_jk].copy()
            if data.empty: continue
            
            # Ambil data terakhir
            last = data.iloc[-1]
            prev = data.iloc[-2]
            
            # Indikator Sederhana
            close = last['Close']
            volume = last['Volume']
            
            # MA Calculation
            ma5 = data['Close'].rolling(5).mean().iloc[-1]
            ma20 = data['Close'].rolling(20).mean().iloc[-1]
            vol_ma20 = data['Volume'].rolling(20).mean().iloc[-1]
            
            # Logika BSJP (Sederhana)
            # 1. Tren Positif (MA5 > MA20)
            # 2. Volume Spike (Volume > Rata-rata Volume)
            # 3. Candle Hijau (Close > Open)
            
            score = 0
            reasons = []
            
            if ma5 > ma20: 
                score += 20
                reasons.append("Uptrend")
            
            if volume > vol_ma20:
                score += 30
                reasons.append("Vol Spike")
                
            if close > last['Open']:
                score += 20
                reasons.append("Green Candle")
                
            # Potensi Gap Up (Close dekat High)
            if close >= last['High'] * 0.99:
                score += 30
                reasons.append("Strong Close")

            if score >= 50: # Hanya tampilkan yang skornya bagus
                results.append({
                    "Ticker": t,
                    "Close": int(close),
                    "Score": score,
                    "Signal": ", ".join(reasons)
                })
                
        except Exception:
            continue
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("="*40)
    print("   BSJP CLI SCREENER (LIVE DATA)   ")
    print("="*40)
    
    # 1. Download
    raw_data = get_data(DEFAULT_TICKERS)
    
    if raw_data is not None and not raw_data.empty:
        # 2. Analyze
        df_res = analyze_bsjp(raw_data, DEFAULT_TICKERS)
        
        # 3. Print Result
        if not df_res.empty:
            df_res = df_res.sort_values("Score", ascending=False)
            print("\n✅ HASIL SCREENING:")
            # Print manual table formatting agar rapi tanpa library tambahan
            print(f"{'Ticker':<8} | {'Close':<8} | {'Score':<5} | {'Signal'}")
            print("-" * 50)
            for index, row in df_res.iterrows():
                print(f"{row['Ticker']:<8} | {row['Close']:<8} | {row['Score']:<5} | {row['Signal']}")
        else:
            print("\n❌ Tidak ada saham yang memenuhi kriteria BSJP hari ini.")
    else:
        print("Gagal mengambil data.")
    
    print("\nDone.")