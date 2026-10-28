"""Phase 0: the corrected panel, with each fix behind a switch so its effect can be shown on its own.
    build(real_price=True, no_cap=True, median_adv=True, next_filter="truncation")  -> monthly panel (ranked, scored set flagged)
Fixes
  real_price : price >= $1 on the price as actually traded (yfinance splits undone; research/data/real_prices.csv.gz) instead of the split-adjusted price
  no_cap     : drop the "adjusted price ever > $5,000" exclusion (it uses the future)
  median_adv : every liquidity threshold ($100k fitting universe, $1m long, $3m short) on the month's MEDIAN daily dollar volume, not the mean
  next_filter: "shipped" = score only rows with >= 10 trading days next month (uses the future);
               "truncation" = score every tradeable row except where the data ends because the Kaggle archive ends (1 Apr 2020), which is
               missing data, not a halt. Training rows still need a realised target (that is past data at fit time).
The panel keeps the shipped column names (adv, tradeable, liquid) so adr_longshort.holdings and backtest.py work unchanged."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "."); import adr_longshort.features as fe
from adr_longshort import load_prices, load_rf, rank_transform, X_COLS
ARCHIVE_END = pd.Timestamp("2020-04-01")
_cache = {}
def _daily_extras():
    if "x" not in _cache:
        R = pd.read_csv("research/data/real_prices.csv.gz", parse_dates=["date"]); R["ym"] = R.date.dt.to_period("M"); R["dv"] = R.close * R.volume
        g = R.groupby(["ticker", "ym"]); X = pd.DataFrame({"price_real": g.close_real.last(), "adv_med": g.dv.median()}).reset_index()
        last = R.groupby("ticker").date.max(); _cache["x"] = (X, last)
    return _cache["x"]
def build(real_price=True, no_cap=True, median_adv=True, next_filter="truncation", P=None):
    fe.MAX_ADJ_PRICE = float("inf") if no_cap else 5000
    M, stats = fe.build_panel(load_prices() if P is None else P); fe.MAX_ADJ_PRICE = 5000
    _, rfm = load_rf(); M = fe.add_target(M, rfm); M = M[M.ym >= "2000-01"].copy()
    X, last = _daily_extras(); M = M.merge(X, on=["ticker", "ym"], how="left")
    M["adv_mean"] = M.adv
    if median_adv: M["adv"] = M.adv_med.fillna(M.adv_mean)
    px = M.price_real.fillna(M.price) if real_price else M.price
    M["tradeable"] = (M.ndays >= fe.MIN_DAYS) & (px >= fe.MIN_PRICE) & (M.adv >= fe.MIN_DOLVOL)
    M["train_ok"] = M.tradeable & M.exret_next.notna() & (M.ndays_next >= 10)
    if next_filter == "shipped": M["score_ok"] = M.train_ok
    else:
        trunc = M.ticker.map(last).eq(ARCHIVE_END) & (M.ym >= "2020-03")
        M["score_ok"] = M.tradeable & ~trunc
    M["in_sample"] = M.score_ok                      # ranks are taken over the scored cross-section, as the shipped refit.py does
    M = M[M.score_ok | M.train_ok].copy(); M = rank_transform(M, universe=M.score_ok | M.train_ok); M["liquid"] = M.adv >= 1e6
    return M, stats
def refit_predict(M, first=2010, last=2026, extra_cols=()):
    """Shipped protocol on the corrected panel: fit on train_ok rows, score every score_ok row of the test year."""
    from adr_longshort import annual_refit, predict
    T = M[M.train_ok]; out = []; hp = []
    for Y in range(first, last + 1):
        te = M[(M.year == Y) & M.score_ok].copy()
        if not len(te): continue
        m, info = annual_refit(T, Y); te["p_RF"] = predict(m, te); out.append(te); hp.append(dict(test_year=Y, **{k: info[k] for k in ("max_depth", "min_samples_leaf", "val_mse", "train_rows")}))
    P = pd.concat(out); return P[["ticker", "ym", "liquid", "adv", "adv_mean", "exret_next", "p_RF", *extra_cols]], pd.DataFrame(hp)
