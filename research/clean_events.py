"""Every day the cleaning changes, joined to the backtest holdings (formed at month-end ym, held through ym+1, $3m short rule).
For each event inside a held position: side, raw daily return, the name's month return raw vs cleaned, and whether the price move persisted."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "."); from adr_longshort import load_prices, load_predictions, clean, holdings
P = load_prices(); P = P.sort_values(["ticker", "date"]).reset_index(drop=True)
P["raw"] = P.groupby("ticker").close.pct_change()
Q, st = clean(P[["ticker", "date", "close", "volume"]]); print("cleaning counts:", st)
X = P.merge(Q[["ticker", "date", "ret"]], on=["ticker", "date"], how="left")        # dropped spike rows -> ret NaN after merge
X["dropped"] = ~X.set_index(["ticker", "date"]).index.isin(Q.set_index(["ticker", "date"]).index)
ev = X[X.raw.notna() & (X.ret.isna() | (X.ret - X.raw).abs() > 1e-9)].copy()
F = load_predictions(); L = F[F.liquid]; H = {}
for ym, g in L.groupby("ym"):
    h = holdings(g)
    if h: H[ym + 1] = h                                                               # month in which the positions are held
ev["hm"] = ev.date.dt.to_period("M")
ev["side"] = [("long" if t in H[m]["long"] else "short" if t in H[m]["short"] else None) if m in H else None for t, m in zip(ev.ticker, ev.hm)]
held = ev[ev.side.notna()].copy()
# month return of that name in the holding month, raw vs cleaned (as the backtest compounds daily returns, NaN -> 0)
def mret(t, m, col):
    s = (Q if col == "ret" else P); s = s[(s.ticker == t) & (s.date.dt.to_period("M") == m)]
    return (1 + s[col].fillna(0)).prod() - 1
held["month_raw"] = [mret(t, m, "raw") for t, m in zip(held.ticker, held.hm)]; held["month_clean"] = [mret(t, m, "ret") for t, m in zip(held.ticker, held.hm)]
def persist(t, d):    # close 10 trading days later / close the day before the event
    s = P[P.ticker == t].set_index("date").close; i = s.index.get_loc(d)
    return s.iloc[min(i + 10, len(s) - 1)] / s.iloc[i - 1] if i > 0 else np.nan
held["px_10d_vs_pre"] = [persist(t, d) for t, d in zip(held.ticker, held.date)]
held["vol_ratio"] = [P[(P.ticker == t) & (P.date == d)].volume.iloc[0] / max(P[(P.ticker == t) & (P.date < d)].volume.tail(20).median(), 1) for t, d in zip(held.ticker, held.date)]
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(f"\nchanged days: {len(ev)} total, {len(held)} inside held positions ({(held.side=='short').sum()} short, {(held.side=='long').sum()} long)")
print(held[["ticker", "date", "side", "close", "raw", "dropped", "month_raw", "month_clean", "px_10d_vs_pre", "vol_ratio"]].round(3).to_string(index=False))
held.to_csv("research/out/clean_events_held.csv", index=False)
