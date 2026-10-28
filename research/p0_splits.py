"""Phase 0.1a: split events from yfinance for every ticker in the Yahoo price file -> research/data/splits_yf.csv (ticker, date, ratio).
ratio = new shares per old share (2.0 = 2-for-1, 0.1 = 1-for-10 reverse)."""
import time, pandas as pd, yfinance as yf
Y = pd.read_csv("data/yf_prices.csv.gz", usecols=["ticker"]).ticker.unique(); rows = []; fail = []
for i, t in enumerate(sorted(Y)):
    for attempt in range(3):
        try:
            s = yf.Ticker(t).splits; break
        except Exception as e:
            s = None; time.sleep(2)
    if s is None: fail.append(t); continue
    for d, r in s.items(): rows.append((t, pd.Timestamp(d).tz_localize(None).normalize(), float(r)))
    if (i + 1) % 50 == 0: print(i + 1, "tickers", len(rows), "splits", flush=True)
S = pd.DataFrame(rows, columns=["ticker", "date", "ratio"]); S.to_csv("research/data/splits_yf.csv", index=False)
print(f"{len(Y)} tickers, {len(S)} split events, {S.ticker.nunique()} tickers with splits, {(S.ratio<1).sum()} reverse; failed: {fail}")
