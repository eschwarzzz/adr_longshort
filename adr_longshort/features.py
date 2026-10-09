"""Monthly characteristics for the Chinese ADR random-forest model, computed from daily adjusted closes and volumes.
Everything at month-end t uses data up to and including t.

Input: a DataFrame with columns ticker, date, close (dividend- and split-adjusted), volume. Output: one row per ticker-month with the 16 raw
characteristics, the tradeable flag and the liquidity measure; rank_transform adds the model inputs (*_r in [-1, 1], missing -> 0)."""
import numpy as np, pandas as pd
from .clean_prices import clean

FEATS = ["ret_1m", "mom_6_1", "mom_12_1", "mom_36_13", "vol_1m", "vol_3m", "ivol_12m", "beta_12m", "maxret_1m", "minret_1m", "dolvol_1m", "dolvol_trend", "illiq_1m", "zero_1m", "high52", "age"]
X_COLS = [f + "_r" for f in FEATS]
MIN_PRICE = 1.0; MIN_DOLVOL = 1e5; MIN_DAYS = 15     # tradeable at the signal date: >= 15 trading days in the month, price >= $1, average daily dollar volume over the month >= $100k
MIN_HISTORY_DAYS = 130                     # a name needs this many daily rows before it enters the panel
MAX_ADJ_PRICE = 5000                       # names whose adjusted price ever exceeds this are serial reverse-splitters and are dropped

def daily(P):
    """Clean the daily prices, add dollar volume and the equal-weight universe return used for betas."""
    P = P[["ticker", "date", "close", "volume"]].dropna(subset=["close"]).sort_values(["ticker", "date"]); P = P[P.close > 0]
    P, stats = clean(P); P["dolvol"] = P.close * P.volume
    idx = P.groupby("date").ret.agg(["mean", "count"]); idx = idx[idx["count"] >= 20]["mean"].rename("mkt")
    P = P.merge(idx, left_on="date", right_index=True, how="left"); P["ym"] = P.date.dt.to_period("M")
    return P, stats

def monthly_one(g):
    """Characteristics for one ticker. g: that ticker's cleaned daily rows."""
    g = g.set_index("date"); m = pd.DataFrame(index=g.ym.unique())
    me = g.groupby("ym").last(); m["price"] = me.close; m["ret_1m"] = g.groupby("ym").ret.apply(lambda x: (1 + x).prod() - 1)
    m["vol_1m"] = g.groupby("ym").ret.std(); m["maxret_1m"] = g.groupby("ym").ret.max(); m["minret_1m"] = g.groupby("ym").ret.min()
    m["dolvol_1m"] = np.log(g.groupby("ym").dolvol.mean().clip(lower=1)); m["zero_1m"] = g.groupby("ym").ret.apply(lambda x: (x.abs() < 1e-8).mean())
    m["illiq_1m"] = np.log(g.groupby("ym").apply(lambda x: (x.ret.abs() / x.dolvol.clip(lower=1)).mean() * 1e6 + 1e-9, include_groups=False))
    m["ndays"] = g.groupby("ym").ret.count()
    mret = m["ret_1m"]
    m["mom_12_1"] = (1 + mret).rolling(11).apply(np.prod, raw=True).shift(1) - 1; m["mom_6_1"] = (1 + mret).rolling(5).apply(np.prod, raw=True).shift(1) - 1
    m["mom_36_13"] = (1 + mret).rolling(24).apply(np.prod, raw=True).shift(12) - 1; m["vol_3m"] = g.ret.groupby(g.ym).std().rolling(3).mean()
    m["dolvol_trend"] = m["dolvol_1m"] - m["dolvol_1m"].rolling(12).mean().shift(1)
    m["high52"] = me.close / g.close.groupby(g.ym).max().rolling(12).max()
    m["age"] = np.log(np.arange(1, len(m) + 1))
    out = []
    for ym in m.index:                       # beta and residual volatility against the universe over the trailing 12 months of daily data
        w = g[(g.ym <= ym) & (g.ym > ym - 12)].dropna(subset=["ret", "mkt"])
        if len(w) < 60: out.append((np.nan, np.nan)); continue
        b = np.cov(w.ret, w.mkt)[0, 1] / np.var(w.mkt); out.append((b, (w.ret - b * w.mkt).std()))
    return m.join(pd.DataFrame(out, index=m.index, columns=["beta_12m", "ivol_12m"]))

def build_panel(P, verbose=False):
    """Daily prices -> monthly panel with raw characteristics, the tradeable flag and average daily dollar volume (adv). Ranks are added separately."""
    P, stats = daily(P); panels = []; tickers = sorted(P.ticker.unique())
    for i, (t, g) in enumerate(P.groupby("ticker")):
        if len(g) < MIN_HISTORY_DAYS: continue
        mm = monthly_one(g.reset_index(drop=True)); mm["ticker"] = t; panels.append(mm)
        if verbose and (i + 1) % 50 == 0: print(f"  {i + 1}/{len(tickers)} names")
    M = pd.concat(panels).rename_axis("ym").reset_index()
    full = M[(M.ndays >= MIN_DAYS) & (M.ym >= "2000-01")]; bad = full.groupby("ticker").price.max(); M = M[~M.ticker.isin(bad[bad > MAX_ADJ_PRICE].index)]
    M["adv"] = np.exp(M.dolvol_1m); M["tradeable"] = (M.ndays >= MIN_DAYS) & (M.price >= MIN_PRICE) & (M.adv >= MIN_DOLVOL); M["year"] = M.ym.dt.year
    return M.sort_values(["ticker", "ym"]).reset_index(drop=True), stats                     # months with fewer than MIN_DAYS trading days stay in the frame (not tradeable) so that targets shift over them as in the research panel

def add_target(M, rf_monthly):
    """Research only: the next row's return (the next month with any trading) in excess of the 1-month bill of the next calendar month, and that row's trading days.
    rf_monthly: Series indexed by Period[M], the bill rate as a decimal per month."""
    M = M.sort_values(["ticker", "ym"]).copy(); g = M.groupby("ticker")
    M["ret_next"] = g.ret_1m.shift(-1); M["ndays_next"] = g.ndays.shift(-1)
    M["rf_next"] = M.ym.map(rf_monthly.shift(-1)); M["exret_next"] = M.ret_next - M.rf_next          # shift on the bill series: a month before the series starts gets no target, as in the research panel
    return M

def research_mask(M):
    """The rows the research panel ranked and fitted on: tradeable, with a next-month return over at least 10 trading days."""
    return M.tradeable & M.exret_next.notna() & (M.ndays_next >= 10)

def rank_transform(M, universe=None):
    """Within-month percentile ranks over `universe` (boolean mask; default = tradeable rows) mapped to [-1, 1]; missing -> 0.
    Ranks must be computed over the same kind of universe the model was fitted on. Rows outside the universe get no ranks (0s) and should not be scored."""
    M = M.copy(); u = M.tradeable if universe is None else universe; T = M[u]
    for f in FEATS: M.loc[T.index, f + "_r"] = T.groupby("ym")[f].rank(pct=True) * 2 - 1
    M[X_COLS] = M[X_COLS].fillna(0.0); return M
