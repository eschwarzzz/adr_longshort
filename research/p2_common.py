"""Phase 2 helpers: refit the forest with any feature subset on the corrected panel, write forecasts, and report with metrics.full_report.
Fixed hyperparameters (research/out/p0_hyper_corrected.csv) for exploratory grids; 'grid' re-chooses depth/leaf on validation MSE each year."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "research")
from sklearn.ensemble import RandomForestRegressor
from adr_longshort import FEATS, holdings
GRID = [(d, l) for d in (2, 4, 6) for l in (20, 50)]
_M = {}
def panel():
    if "M" not in _M:
        M = pd.read_csv("research/out/p0_panel_corrected.csv"); M["ym"] = pd.PeriodIndex(M.ym, freq="M"); _M["M"] = M
        _M["H"] = pd.read_csv("research/out/p0_hyper_corrected.csv").set_index("test_year")
    return _M["M"], _M["H"]
def _rf(d, l, seed): return RandomForestRegressor(n_estimators=300, max_depth=d, min_samples_leaf=l, max_features=0.5, n_jobs=-1, random_state=seed)
def val_spread(va, p):
    """Validation long-short (liquid names, $3m short floor) mean monthly spread, and mean monthly rank IC."""
    v = va.assign(p=p); ls = []; ic = []
    for ym, g in v.groupby("ym"):
        ic.append(g.p.rank().corr(g.exret_next.rank()))
        h = holdings(g[g.liquid], "p")
        if h: ls.append(g[g.ticker.isin(h["long"])].exret_next.mean() - g[g.ticker.isin(h["short"])].exret_next.mean())
    return float(np.mean(ls)) if ls else np.nan, float(np.nanmean(ic))
def fit_year(M, Y, cols, hp="fixed", seed=0, H=None, want_val=False):
    T = M[M.train_ok]; tr = T[(T.year >= 2001) & (T.year <= Y - 3)]; va = T[(T.year >= Y - 2) & (T.year <= Y - 1)]; te = M[(M.year == Y) & M.score_ok]
    lo, hi = tr.exret_next.quantile([.01, .99]); y = tr.exret_next.clip(lo, hi).values; yv = va.exret_next.clip(lo, hi).values
    if hp == "fixed": d, l = int(H.loc[Y, "max_depth"]), int(H.loc[Y, "min_samples_leaf"]); m = _rf(d, l, seed).fit(tr[cols].values, y)
    else:
        best = None
        for d, l in (hp if isinstance(hp, list) else GRID):
            mm = _rf(d, l, seed).fit(tr[cols].values, y); mse = float(((yv - mm.predict(va[cols].values)) ** 2).mean())
            if best is None or mse < best[0]: best = (mse, d, l, mm)
        _, d, l, m = best
    p = pd.Series(m.predict(te[cols].values), index=te.index); info = dict(Y=Y, depth=d, leaf=l)
    if want_val: info["val_ls"], info["val_ic"] = val_spread(va, m.predict(va[cols].values))
    return p, info
def fit_variant(feats, hp="fixed", seed=0, hyper=None, want_val=False):
    """feats: raw feature names. hyper: dict Y->(depth, leaf) to reuse a grid choice with another seed. Returns (pred Series over score_ok rows 2010-26, info DataFrame)."""
    M, H = panel(); cols = [f + "_r" for f in feats]; ps = []; infos = []
    for Y in range(2010, 2027):
        if hyper is not None: p, i = fit_year(M, Y, cols, hp=[hyper[Y]], seed=seed, want_val=want_val)
        else: p, i = fit_year(M, Y, cols, hp=hp, seed=seed, H=H, want_val=want_val)
        ps.append(p); infos.append(i)
    return pd.concat(ps), pd.DataFrame(infos).set_index("Y")
def write_preds(cols, path):
    """cols: dict name -> prediction Series (index = panel rows). Writes ticker, ym, liquid, adv, adv_mean, exret_next + columns."""
    M, _ = panel(); idx = next(iter(cols.values())).index; out = M.loc[idx, ["ticker", "ym", "liquid", "adv", "adv_mean", "exret_next"]].copy()
    for k, v in cols.items(): out[k] = v.reindex(idx)
    out.to_csv(path, index=False); return path
def report(path, cols, account=True):
    from metrics import full_report
    rows = []
    for c in cols:
        o, _ = full_report(path, c, account=account); o["variant"] = c; rows.append(o)
    return pd.DataFrame(rows).set_index("variant")
KEEP = ["unit_ret", "unit_sharpe", "unit_net_sharpe", "unit_net_harsh", "long_ret", "short_ret", "beta_pgj", "beta_vw", "unit_maxdd", "unit_sharpe_2010_15", "unit_sharpe_2016_20", "unit_sharpe_2021_26",
        "paired_t_vs_base", "acct_final_equity", "acct_cagr_pct", "acct_sharpe", "acct_maxdd_pct", "acct_worst_month_pct", "acct_beta_pgj", "acct_sharpe_2021_26"]
