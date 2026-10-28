"""Phase 4.1b: calibrate the Abdi-Ranaldo spread against live quoted bid/ask (yfinance info) for the latest liquid names."""
import sys, time, numpy as np, pandas as pd, yfinance as yf
sys.path.insert(0, ".")
SP = pd.read_csv("research/out/p4_spreads.csv"); last = SP.ym.max(); prev = sorted(SP.ym.unique())[-2]
PR = pd.read_csv("research/out/p0_pred_corrected.csv"); liq = PR[(PR.ym == PR.ym.max()) & PR.liquid].ticker.unique()
rows = []
for t in liq:
    try: i = yf.Ticker(t).info; b, a = i.get("bid"), i.get("ask"); px = i.get("regularMarketPrice") or i.get("previousClose")
    except Exception: continue
    if b and a and a > b > 0: rows.append(dict(ticker=t, bid=b, ask=a, px=px, quoted_bp=(a - b) / ((a + b) / 2) * 1e4))
    time.sleep(0.3)
Q = pd.DataFrame(rows); X = Q.merge(SP[SP.ym.isin([last, prev])].groupby("ticker").agg(ar_bp=("ar_spread", lambda s: s.mean() * 1e4), px_real=("px_real", "last")), on="ticker")
X["ratio"] = X.ar_bp / X.quoted_bp; X.to_csv("research/out/p4_spread_calibration.csv", index=False)
print(f"names with live quotes: {len(X)} of {len(liq)}"); print(X[["quoted_bp", "ar_bp", "ratio"]].describe(percentiles=[.25, .5, .75]).round(2).to_string())
print("by price bucket (median quoted vs AR, bp):"); X["pxb"] = pd.cut(X.px, [0, 2, 5, 20, 1e9], labels=["<$2", "$2-5", "$5-20", ">$20"]); print(X.groupby("pxb", observed=True)[["quoted_bp", "ar_bp"]].median().round(1).to_string())
