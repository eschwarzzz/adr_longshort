"""Phase 5 pipeline: the repo's frozen pipeline with the three corrections, applied identically to every universe.
Features, cleaning, ranks, refit protocol and quintile rule are the package's own (adr_longshort.*). Corrections:
  (1) liquidity floors on the month's MEDIAN daily dollar volume, not the mean;
  (2) no exclusion of names whose adjusted price ever exceeds $5,000;
  (3) the >= $1 price test on the price as actually traded (Yahoo split events undo the split adjustment).
Everything else (MIN_HISTORY_DAYS, the next-month >= 10 days research mask, the grid, the floors) is unchanged."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import adr_longshort.features as fe
from adr_longshort import annual_refit, predict, rank_transform, holdings, load_rf, X_COLS
from scipy.stats import spearmanr
D = "research/data/p5"; BASE = [(25e6, .01), (10e6, .04), (3e6, .12), (0, .35)]; COST = 0.002
_, RFM = load_rf()

def split_factor(P, S):
    """Per daily row: product of split ratios dated after the row (adjusted close x factor = price as traded)."""
    f = pd.Series(1.0, index=P.index)
    if S is None or not len(S): return f
    S = S.copy(); S["date"] = pd.to_datetime(S.date).dt.tz_localize(None)
    for t, s in S.groupby("ticker"):
        idx = P.index[P.ticker == t]
        if not len(idx): continue
        s = s.sort_values("date"); cum = np.r_[np.cumprod(s.ratio.values[::-1])[::-1], 1.0]       # cum[k] = product of ratios k.. end
        pos = np.searchsorted(s.date.values, P.loc[idx, "date"].values, side="right"); f.loc[idx] = cum[pos]
    return f

def build(P, S=None, fx=1.0, min_price=1.0):
    """Daily prices (ticker,date,close,volume) -> corrected research panel with exret_next and in_sample."""
    P = P.dropna(subset=["close"]); P = P[P.close > 0].copy(); P["date"] = pd.to_datetime(P.date).dt.tz_localize(None); P = P.sort_values(["ticker", "date"]).reset_index(drop=True)
    fe.MAX_ADJ_PRICE = float("inf")
    M, st = fe.build_panel(P)
    P["dv"] = P.close * P.volume; P["traded"] = P.close * split_factor(P, S); P["ym"] = P.date.dt.to_period("M")
    m = P.groupby(["ticker", "ym"]).agg(med_dv=("dv", "median"), px_traded=("traded", "last")).reset_index()
    M = M.merge(m, on=["ticker", "ym"], how="left"); M["adv_mean"] = M.adv; M["adv"] = M.med_dv / fx
    M["tradeable"] = (M.ndays >= fe.MIN_DAYS) & (M.px_traded >= min_price) & (M.adv >= fe.MIN_DOLVOL)
    M = fe.add_target(M, RFM); M["in_sample"] = fe.research_mask(M); M = M[M.ym >= "2000-01"]
    return M, st

def fit_predict(M, first_test=2010, last_test=2026, min_train=1500, min_val=300, models=None):
    """Annual refits with the repo protocol. Returns (scored in-sample rows with p_RF, {year: model})."""
    R = rank_transform(M[M.in_sample].copy()); R["liquid"] = R.adv >= 1e6; out = []; fitted = {}
    for Y in range(first_test, last_test + 1):
        te = R[R.year == Y].copy()
        if not len(te): continue
        if models is not None:
            if Y not in models: continue
            te["p_RF"] = predict(models[Y], te); out.append(te); continue
        ntr = ((R.year >= 2001) & (R.year <= Y - 3)).sum(); nva = ((R.year >= Y - 2) & (R.year <= Y - 1)).sum()
        if ntr < min_train or nva < min_val: continue
        m, info = annual_refit(R, Y); fitted[Y] = m; te["p_RF"] = predict(m, te); out.append(te)
    return (pd.concat(out) if out else R.iloc[:0].assign(p_RF=np.nan)), fitted

def evaluate(F, bench=None, col="p_RF", label=""):
    """Quintile rule (>= $1m long, >= $3m short, 25+ liquid names) plus an unrestricted quintile sort over all tradeable rows."""
    sr = lambda x: x.mean() / x.std() * np.sqrt(12) if len(x) > 2 and x.std() > 0 else np.nan
    tier = lambda a: next(r for lo, r in BASE if a >= lo)
    rows = []; p5 = p1 = set()
    for ym, g in F[F.liquid].groupby("ym"):
        h = holdings(g, col)
        if h is None: continue
        lo = g[g.ticker.isin(h["long"])]; sh = g[g.ticker.isin(h["short"])]; s5, s1 = set(h["long"]), set(h["short"])
        rows.append(dict(ym=ym, L=lo.exret_next.mean(), S=sh.exret_next.mean(), EW=g.exret_next.mean(), nliq=len(g), nS=len(sh),
                         to5=1 - len(s5 & p5) / len(s5) if p5 else 1, to1=1 - len(s1 & p1) / len(s1) if p1 else 1, bb=np.mean([tier(a) for a in sh.adv]) / 12))
        p5, p1 = s5, s1
    q = pd.DataFrame(rows).set_index("ym") if rows else pd.DataFrame()
    allq = []
    for ym, g in F.groupby("ym"):
        if len(g) < 25: continue
        k = pd.qcut(g[col].rank(method="first"), 5, labels=False); r = g.groupby(k).exret_next.mean(); allq.append((ym, r.get(4, np.nan) - r.get(0, np.nan), r.get(0, np.nan), g.exret_next.mean()))
    A = pd.DataFrame(allq, columns=["ym", "ls_all", "q1_all", "ew_all"]).set_index("ym")
    ic = F.groupby("ym").apply(lambda g: spearmanr(g[col], g.exret_next).statistic if len(g) >= 10 else np.nan, include_groups=False).dropna()
    d = dict(universe=label, months_rule=len(q), first=str(F.ym.min()) if len(F) else "", names_tradeable=F.groupby("ym").size().mean() if len(F) else np.nan,
             names_liquid=q.nliq.mean() if len(q) else np.nan, n_short=q.nS.mean() if len(q) else np.nan, rank_ic=ic.mean(), ic_t=ic.mean() / ic.std() * np.sqrt(len(ic)) if len(ic) > 2 else np.nan,
             months_all=len(A), ls_all_ret=A.ls_all.mean() * 1200, ls_all_sharpe=sr(A.ls_all), q1_all_minus_ew=(A.q1_all - A.ew_all).mean() * 1200)
    if len(q):
        ls = q.L - q.S; net = ls - COST * 2 * (q.to5 + q.to1) - q.bb
        d.update(ls_ret=ls.mean() * 1200, ls_sharpe=sr(ls), net_sharpe=sr(net), short_alone_sharpe=sr(-q.S), short_alone_ret=-q.S.mean() * 1200, long_ew_sharpe=sr(q.L - q.EW), long_ew_ret=(q.L - q.EW).mean() * 1200)
        for lab, (a, b) in {"10_15": ("2010", "2015"), "16_20": ("2016", "2020"), "21_26": ("2021", "2026")}.items():
            x = ls.loc[a:b] if len(ls.loc[a:b]) else ls.iloc[:0]; d[f"ls_sh_{lab}"] = sr(x) if len(x) >= 12 else np.nan
        if bench is not None:
            b = bench.reindex(q.index + 1).values; ok = ~np.isnan(b); d["beta_ls"] = np.cov(ls[ok], b[ok])[0, 1] / np.var(b[ok]) if ok.sum() > 12 else np.nan
            d["beta_short_leg"] = np.cov(q.S[ok], b[ok])[0, 1] / np.var(b[ok]) if ok.sum() > 12 else np.nan
    for lab, (a, b) in {"10_15": ("2010", "2015"), "16_20": ("2016", "2020"), "21_26": ("2021", "2026")}.items():
        x = A.ls_all.loc[a:b] if len(A) else A; d[f"all_sh_{lab}"] = sr(x.dropna()) if len(x) >= 12 else np.nan
    return d, q, A

def bench_monthly(tk):
    B = pd.read_csv(f"{D}/bench_prices.csv.gz", parse_dates=["date"]); b = B[B.ticker == tk].set_index("date").close; b.index = pd.to_datetime(b.index).tz_localize(None)
    m = b.resample("ME").last().pct_change(); m.index = m.index.to_period("M"); return (m - RFM.reindex(m.index)).dropna()

def log_trial(trial_id, desc, metric, file):
    import os, csv; p = "research/out/trial_log_p5.csv"; new = not os.path.exists(p)
    with open(p, "a", newline="") as fh:
        w = csv.writer(fh)
        if new: w.writerow(["trial_id", "phase", "description", "metric_summary", "file"])
        w.writerow([trial_id, "5", desc, metric, file])
