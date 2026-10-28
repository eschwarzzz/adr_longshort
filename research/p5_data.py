"""Phase 5 data: universe lists and Yahoo daily prices + split events for the portability tests. Free sources only.
    python research/p5_data.py [china_splits|gc_ext|foreign|us|hk|bench|all]
Writes research/data/p5/<name>_prices.csv.gz (ticker,date,close,volume; split- and dividend-adjusted closes, split-adjusted volume)
and <name>_splits.csv (ticker,date,ratio; Yahoo convention: 2.0 = 2-for-1, 0.1 = 1-for-10 reverse)."""
import sys, os, time, re, numpy as np, pandas as pd, yfinance as yf
D = "research/data/p5"; os.makedirs(D, exist_ok=True); SEED = 7

def clean_symbols(s):
    """Common shares only: drop warrants, units, rights, preferreds and odd symbols."""
    s = s[~s.symbol.str.contains(r"[\^/ ]", regex=True)]
    return s[~((s.symbol.str.len() >= 5) & s.symbol.str[-1].isin(list("WURP")))]

def universes():
    S = clean_symbols(pd.read_csv(f"{D}/nasdaq_screener.csv")); U = pd.read_csv("data/universe.csv"); Y = pd.read_csv("data/yf_prices.csv.gz", usecols=["ticker"]).ticker.unique()
    gc = S[S.country.isin(["China", "Hong Kong", "Macau"]) & ~S.symbol.isin(U.ticker)]
    asia = ["Singapore", "Malaysia", "Japan", "Taiwan", "Thailand", "Indonesia", "Philippines", "Vietnam", "South Korea", "Cayman Islands", "British Virgin Islands"]
    fo = S[S.country.isin(asia) & (S.marketCap.fillna(0) < 2e9)]                     # micro and small foreign listings; excludes the large ADRs (TSM, Sony, Sea ...)
    us = S[(S.country == "United States") & (S.marketCap.fillna(0) > 0) & (S.marketCap < 1e9)].sample(800, random_state=SEED)
    X = pd.read_excel(f"{D}/hkex_list.xlsx", header=2); X = X[(X.Category == "Equity") & X["Sub-Category"].isin(["Equity Securities (Main Board)", "Equity Securities (GEM)"]) & (X["Trading Currency"] == "HKD")]
    X["yf"] = X["Stock Code"].astype(int).map(lambda c: f"{c:04d}.HK"); hk = X.sample(1000, random_state=SEED)
    return dict(china_splits=list(Y), gc_ext=list(gc.symbol), foreign=list(fo.symbol), us=list(us.symbol), hk=list(hk.yf),
                bench=["PGJ", "KWEB", "EEM", "IWC", "IWM", "EWH", "^HSI", "2800.HK", "SPY"]), dict(gc=gc, fo=fo, us=us, hk=hk)

def fetch(tickers, name, chunk=100):
    out, spl = [], []
    for i in range(0, len(tickers), chunk):
        tk = tickers[i:i + chunk]; df = None
        for attempt in range(3):
            try:
                df = yf.download(tk, start="1999-01-01", auto_adjust=True, actions=True, group_by="ticker", threads=True, progress=False); break
            except Exception as e: print("retry", repr(e)[:200], flush=True); time.sleep(10)
        if df is None:                                                      # batch failed: one ticker at a time
            parts = {}
            for t in tk:
                try: parts[t] = yf.download(t, start="1999-01-01", auto_adjust=True, actions=True, progress=False, multi_level_index=False)
                except Exception as e: print("skip", t, repr(e)[:120], flush=True)
            df = pd.concat(parts, axis=1) if parts else pd.DataFrame()
        for t in tk:
            try: g = df[t] if isinstance(df.columns, pd.MultiIndex) else df
            except KeyError: continue
            g = g.dropna(subset=["Close"])
            if not len(g): continue
            out.append(pd.DataFrame({"ticker": t, "date": g.index, "close": g.Close.values, "volume": g.Volume.values}))
            if "Stock Splits" in g:
                s = g["Stock Splits"]; s = s[s > 0]; spl += [(t, d, r) for d, r in s.items()]
        print(f"{name}: {min(i + chunk, len(tickers))}/{len(tickers)}", flush=True); time.sleep(2)
    P = pd.concat(out); P.to_csv(f"{D}/{name}_prices.csv.gz", index=False); pd.DataFrame(spl, columns=["ticker", "date", "ratio"]).to_csv(f"{D}/{name}_splits.csv", index=False)
    print(f"{name}: {P.ticker.nunique()} names with prices, {len(spl)} split events", flush=True)

if __name__ == "__main__":
    L, meta = universes(); which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for k, v in meta.items(): v.to_csv(f"{D}/universe_{k}.csv", index=False)
    for k in (L if which == "all" else [which]):
        if os.path.exists(f"{D}/{k}_prices.csv.gz"): print(k, "exists, skipped"); continue
        fetch(L[k], k)
