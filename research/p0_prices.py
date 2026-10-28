"""Phase 0.1b: real traded prices. For each yfinance split, check whether Yahoo's adjusted series actually absorbed it
(adjusted: no jump at the split date; unadjusted: a jump of ~1/ratio). Real price = adjusted close x product of the ratios of
the absorbed splits after that date. Also infer splits Yahoo never listed: a jump of >= 3x either way that persists 10 days with
volume moving the opposite way. Writes research/data/real_prices.csv.gz (ticker, date, close, close_real, volume) and
research/out/p0_split_audit.csv."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "."); from adr_longshort import load_prices
P = load_prices().sort_values(["ticker", "date"]).reset_index(drop=True)
S = pd.read_csv("research/data/splits_yf.csv", parse_dates=["date"])
P["lr"] = np.log(P.close / P.groupby("ticker").close.shift(1))
audit = []
for t, g in S.groupby("ticker"):
    px = P[P.ticker == t].set_index("date")
    for _, s in g.iterrows():
        w = px.loc[s.date - pd.Timedelta(days=4): s.date + pd.Timedelta(days=4)]
        if not len(w): audit.append(dict(ticker=t, date=s.date, ratio=s.ratio, status="no prices near date", jump=np.nan)); continue
        j = w.lr.loc[w.lr.abs().idxmax()] if w.lr.notna().any() else np.nan; expect = np.log(1 / s.ratio)
        status = "unadjusted in Yahoo" if abs(expect) > np.log(1.5) and abs(j - expect) < 0.35 else "absorbed"
        audit.append(dict(ticker=t, date=s.date, ratio=s.ratio, status=status, jump=np.exp(j)))
A = pd.DataFrame(audit)
# real price factor from absorbed splits after each date
P["factor"] = 1.0
for t, g in A[A.status == "absorbed"].groupby("ticker"):
    idx = P.ticker == t; d = P.loc[idx, "date"]; f = np.ones(idx.sum())
    for _, s in g.iterrows(): f *= np.where(d.values < np.datetime64(s.date), s.ratio, 1.0)
    P.loc[idx, "factor"] = f
P["close_real"] = P.close * P.factor
# inferred unlisted splits: >=3x jump either way, level persists 10 days, volume moves opposite
P["c10"] = P.groupby("ticker").close.shift(-10); P["c_1"] = P.groupby("ticker").close.shift(1)
P["v_pre"] = P.groupby("ticker").volume.transform(lambda v: v.shift(1).rolling(20, min_periods=5).median()); P["v_post"] = P.groupby("ticker").volume.transform(lambda v: v.shift(-10).rolling(10, min_periods=5).median())
jump = P.lr.abs() > np.log(3); persist = (np.log(P.c10 / P.c_1) * np.sign(P.lr)) > np.log(3) * 0.6
volopp = (np.log(P.v_post / P.v_pre) * np.sign(P.lr)) < -np.log(2)
inf = P[jump & persist & volopp][["ticker", "date", "close", "c_1", "c10", "v_pre", "v_post"]].copy(); inf["ratio_implied"] = inf.c_1 / inf.close
listed = A.set_index(["ticker", "date"]).index
inf["matches_listed_split"] = [any((A.ticker == t) & ((A.date - d).abs() <= pd.Timedelta(days=5))) for t, d in zip(inf.ticker, inf.date)]
A.to_csv("research/out/p0_split_audit.csv", index=False); inf.to_csv("research/out/p0_split_inferred.csv", index=False)
P[["ticker", "date", "close", "close_real", "volume"]].to_csv("research/data/real_prices.csv.gz", index=False)
print("yfinance splits:", A.status.value_counts().to_dict())
print(f"inferred split-like jumps: {len(inf)} ({inf.matches_listed_split.sum()} match a listed split, {(~inf.matches_listed_split).sum()} unlisted)")
print(inf[~inf.matches_listed_split].round(2).to_string(index=False)[:3000])
print("\nGMM:", A[A.ticker == "GMM"].to_string(index=False))
m = P.groupby("ticker").close.max(); r = P.groupby("ticker").close_real.max()
print(f"names with adjusted max > $5,000: {(m > 5000).sum()}; with REAL max > $5,000: {(r > 5000).sum()} -> {sorted(r[r > 5000].index)}")
