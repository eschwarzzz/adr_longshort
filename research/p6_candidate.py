"""Phase 6: the pre-specified candidate (research/out/p6_candidate_spec.md).
  1. forecasts: fully corrected panel (p0_panel_corrected_reuse.csv), random forest with the 11 features, shipped protocol and grid
  2. P&L returns with only verified errors removed (p0_audit_events.csv)
  3. backtest.py in full (variants, bootstrap, random-sort null, stale curve, five-year account) with outputs to research/out/p6_*
Usage: python research/p6_candidate.py [features: all|drop5]"""
import sys, re, time, numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "research")
from sklearn.ensemble import RandomForestRegressor
from adr_longshort import X_COLS, load_prices
from adr_longshort.rf_model import GRID, N_TREES, MAX_FEATURES, SEED
from trials import log
mode = sys.argv[1] if len(sys.argv) > 1 else "drop5"; TAG = f"p6_{mode}"
DROP = {"beta_12m_r", "minret_1m_r", "illiq_1m_r", "maxret_1m_r", "age_r"} if mode == "drop5" else set()
COLS = [c for c in X_COLS if c not in DROP]
M = pd.read_csv("research/out/p0_panel_corrected_reuse.csv"); M["ym"] = pd.PeriodIndex(M.ym, freq="M"); T = M[M.train_ok]; out = []; hp = []
t0 = time.time()
for Y in range(2010, 2027):
    te = M[(M.year == Y) & M.score_ok].copy()
    if not len(te): continue
    tr = T[(T.year >= 2001) & (T.year <= Y - 3)]; va = T[(T.year >= Y - 2) & (T.year <= Y - 1)]
    lo, hi = tr.exret_next.quantile([.01, .99]); ytr = tr.exret_next.clip(lo, hi).values; yva = va.exret_next.clip(lo, hi).values
    best = None
    for d, l in GRID:
        m = RandomForestRegressor(n_estimators=N_TREES, max_depth=d, min_samples_leaf=l, max_features=MAX_FEATURES, n_jobs=-1, random_state=SEED).fit(tr[COLS].values, ytr)
        mse = float(((yva - m.predict(va[COLS].values)) ** 2).mean())
        if best is None or mse < best[0]: best = (mse, m, d, l)
    te["p_RF"] = best[1].predict(te[COLS].values); out.append(te); hp.append(dict(test_year=Y, max_depth=best[2], min_samples_leaf=best[3], val_mse=best[0]))
P = pd.concat(out)[["ticker", "ym", "liquid", "adv", "adv_mean", "exret_next", "p_RF"]]; pred = f"research/out/{TAG}_pred.csv"; P.to_csv(pred, index=False)
pd.DataFrame(hp).to_csv(f"research/out/{TAG}_hyper.csv", index=False); print(f"forecasts [{time.time()-t0:.0f}s]", flush=True)
# P&L returns: only verified errors removed
ev = pd.read_csv("research/out/p0_audit_events.csv", parse_dates=["date"]); err = ev[ev.verdict == "verified error"]
drop = set(zip(err[err["filter"].str.startswith("R1")].ticker, err[err["filter"].str.startswith("R1")].date)); nullk = set(zip(err[~err["filter"].str.startswith("R1")].ticker, err[~err["filter"].str.startswith("R1")].date))
L = load_prices().sort_values(["ticker", "date"]); L = L[[(t, d) not in drop for t, d in zip(L.ticker, L.date)]].copy(); L["ret"] = L.groupby("ticker").close.pct_change()
L.loc[[(t, d) in nullk for t, d in zip(L.ticker, L.date)], "ret"] = np.nan
L[L.date >= "2009-12-01"].pivot(index="date", columns="ticker", values="ret").to_pickle(f"research/out/{TAG}_ret.pkl")
# full backtest.py with redirects
s = open("backtest.py").read()
s = s.replace('R = f"{ROOT}/results"', 'R = "research/out"').replace('TAG = "bt500k_rf"', f'TAG = "{TAG}"').replace('QUICK = "--quick" in sys.argv', "QUICK = False")
s = s.replace("F = load_predictions()", f"F = load_predictions({pred!r})")
s = s.replace('P = load_prices(); P, cstats = clean(P); P = P[P.date >= "2009-12-01"]; RET = P.pivot(index="date", columns="ticker", values="ret"); DAYS = RET.index',
              f'RET = pd.read_pickle({f"research/out/{TAG}_ret.pkl"!r}); DAYS = RET.index; cstats = {{}}')
s = s.replace('S1.to_csv(f"{R}/quintile_stats_liquid_ew.csv"); QR[MODEL].reset_index().assign(strategy=MODEL).to_csv(f"{R}/quintile_returns_rf.csv", index=False)', f'S1.to_csv(f"{{R}}/{TAG}_quintile_stats.csv")')
s = s.replace('S2.to_csv(f"{R}/borrow_variants_rf.csv")', f'S2.to_csv(f"{{R}}/{TAG}_borrow_variants.csv")')
s = s.replace('pd.read_csv(f"{R}/research_quintile_returns.csv")', 'pd.read_csv("results/research_quintile_returns.csv")')
assert f"{TAG}_ret.pkl" in s and f"{TAG}_quintile_stats" in s
open(f"research/out/{TAG}_backtest_src.py", "w").write(s)
sys.argv = ["backtest"]; ns = {"__name__": "__main__"}; exec(compile(s, f"{TAG}_backtest", "exec"), ns)
S = ns["S"]; h = S.loc["Headline: $3m short leg, 10% vol target"]
log("P6", f"candidate {mode}: full backtest", f"sharpe {h['sharpe']:.2f} monthly {h['sharpe_monthly']:.2f} final {h['final_equity']:,.0f}", f"research/out/{TAG}_variants.csv")
