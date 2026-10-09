"""Refresh the data under data/ from public sources. The repository ships a snapshot (see data/README.md); run this to extend it.

    python3 fetch_data.py            # Yahoo prices for the universe, SOE extensions through Hong Kong, FRED T-bill, ETF benchmarks
    python3 fetch_data.py --test     # a handful of tickers, written to results/_fetch_test/, to check the plumbing

Sources: Yahoo Finance via yfinance (adjusted closes and volume; currently listed names only), FRED series DTB4WK (1-month T-bill), the
Kaggle "stock-market-dataset" archive (CC0) for names that delisted before April 2020, shipped as data/kaggle_patch/stocks/*.csv and not
refreshed here. Twelve SOE ADRs that delisted in 2021-23 are extended from the end of the Kaggle archive to their delisting dates with the
daily returns of their Hong Kong lines (HKD is pegged), dollar volume carried forward from the ADR's last 60 days scaled by the HK line's volume."""
import os, sys, io, time, subprocess, numpy as np, pandas as pd
from adr_longshort import ROOT
TEST = "--test" in sys.argv; D = f"{ROOT}/data" if not TEST else f"{ROOT}/results/_fetch_test"; os.makedirs(f"{D}/kaggle_patch/stocks", exist_ok=True)
REUSED = ["ACH", "ATAI", "BITA"]                                 # Yahoo tickers now belonging to other companies: never fetched, the Kaggle histories are used
SOE = {"PTR": ("0857.HK", "2022-09-02"), "SNP": ("0386.HK", "2022-09-02"), "LFC": ("2628.HK", "2022-09-02"), "SHI": ("0338.HK", "2022-09-02"), "ACH": ("2600.HK", "2022-09-02"), "GSH": ("0525.HK", "2022-09-02"),
       "HNP": ("0902.HK", "2022-11-30"), "CEA": ("0670.HK", "2023-02-28"), "ZNH": ("1055.HK", "2023-02-28"), "CHL": ("0941.HK", "2021-01-08"), "CHU": ("0762.HK", "2021-01-08"), "CEO": ("0883.HK", "2021-10-15")}
ETFS = ["FXI", "KWEB", "MCHI", "PGJ", "SPY"]

def yahoo(tickers, start="1999-01-01"):
    """Daily adjusted OHLC and volume, long format (ticker, date, Open, High, Low, Close, Volume)."""
    import yfinance as yf
    out = []
    for i in range(0, len(tickers), 50):
        chunk = tickers[i:i + 50]; df = yf.download(chunk, start=start, auto_adjust=True, progress=False, group_by="ticker", threads=True)
        for t in chunk:
            try: x = df[t].dropna(subset=["Close"]) if len(chunk) > 1 else df.dropna(subset=["Close"])
            except KeyError: continue
            if len(x) == 0: continue
            x = x.reset_index().rename(columns={"Date": "date"}); x.insert(0, "ticker", t); out.append(x[["ticker", "date", "Open", "High", "Low", "Close", "Volume"]])
        time.sleep(1)
    return pd.concat(out) if out else pd.DataFrame(columns=["ticker", "date", "Open", "High", "Low", "Close", "Volume"])

def fred(series="DTB4WK"):
    """FRED csv via curl (pandas' own fetch fails certificate checks on some machines)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"; r = subprocess.run(["curl", "-sL", url], capture_output=True, text=True, timeout=60)
    df = pd.read_csv(io.StringIO(r.stdout)); assert "observation_date" in df.columns, "FRED download failed"; return df

def extend_soes(hk):
    """Carry each SOE's Kaggle series to its delisting date with the HK line's returns; the shipped Kaggle files already contain this extension."""
    for t, (line, end) in SOE.items():
        f = f"{ROOT}/data/kaggle_patch/stocks/{t}.csv"
        if not os.path.exists(f): continue
        a = pd.read_csv(f, parse_dates=["Date"]).sort_values("Date"); a = a[a["Adj Close"] > 0]
        if a.Date.max() >= pd.Timestamp(end) - pd.Timedelta(days=7): continue            # already extended
        last = a.iloc[-1]; h = hk[(hk.ticker == line) & (hk.date > last.Date) & (hk.date <= end)].sort_values("date"); h = h[h.Close > 0]
        base = hk[(hk.ticker == line) & (hk.date <= last.Date)].sort_values("date")
        if len(h) < 5 or len(base) == 0: continue
        ret = h.Close.values / base.iloc[-1].Close - 1; adr_dv = (a["Adj Close"] * a["Volume"]).tail(60).mean(); hk_dv = (base.tail(60).Close * base.tail(60).Volume).mean()
        px = last["Adj Close"] * (1 + ret); dv = adr_dv * (h.Close.values * h.Volume.values) / max(hk_dv, 1); vol = np.where(px > 0, dv / px, 0)
        ext = pd.DataFrame({"Date": h.date.values, "Open": px, "High": px, "Low": px, "Close": px, "Adj Close": px, "Volume": vol})
        pd.concat([a, ext]).to_csv(f if not TEST else f"{D}/kaggle_patch/stocks/{t}.csv", index=False); print(f"  {t}: extended {last.Date.date()} -> {h.date.iloc[-1].date()} via {line}")

U = pd.read_csv(f"{ROOT}/data/universe.csv"); tickers = sorted(set(U.ticker) - set(REUSED))
kaggle = {os.path.basename(f)[:-4] for f in os.listdir(f"{ROOT}/data/kaggle_patch/stocks")}; tickers = [t for t in tickers if t not in kaggle]
if TEST: tickers = ["BABA", "BIDU", "JD", "NIO", "YUMC"]
print(f"Yahoo: {len(tickers)} tickers"); Y = yahoo(tickers); print(f"  {Y.ticker.nunique()} names, {len(Y):,} rows, last date {Y.date.max().date()}")
Y.to_csv(f"{D}/yf_prices.csv.gz", index=False, compression="gzip")
print("Hong Kong lines for the SOE extension"); H = yahoo(sorted({v[0] for v in SOE.values()}), start="2019-01-01").rename(columns={"date": "date"}); H["date"] = pd.to_datetime(H.date)
extend_soes(H)
print("FRED DTB4WK"); fred("DTB4WK").to_csv(f"{D}/fred_DTB4WK.csv", index=False)
print("ETF benchmarks"); E = yahoo(ETFS, start="1993-01-01"); E["ym"] = pd.to_datetime(E.date).dt.to_period("M")
B = E.groupby(["ticker", "ym"]).Close.last().unstack(0).pct_change(); B.index.name = "Date"; B.to_csv(f"{D}/benchmarks_monthly.csv")
print("written to", D)
