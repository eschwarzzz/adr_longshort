"""Phase 1.2/1.3: out-of-sample forecast paths, test years 2008-2026, on the corrected panel, for every refit procedure and model.
Each procedure is a function of the data available at the end of year E = Y-1 (rows with ym <= Dec E; the Dec-E row's target is
January Y, known before the first use of the model at the end of January Y, as in the shipped protocol).
  A   shipped freeze: train 2001..E-2, choose depth/leaf on E-1..E by MSE, use the model fitted on 2001..E-2
  B   1-year validation: train 2001..E-1, choose on E
  C   retrain through E with A's chosen depth/leaf
  D6  rolling 6-year training window (E-7..E-2), choose on E-1..E
  D8  rolling 8-year window (E-9..E-2), choose on E-1..E
  R   recency-weighted: as A with sample weights 0.5 ** ((E-2 - year) / 3)
  Q   quarterly: A's December choice of depth/leaf, refitted at each quarter start on every train row before it (Q1 = C)
  HUB Huber (epsilon 1.35, alpha 0) on A's training window; OLS likewise
Writes research/out/p1_paths.csv (ticker, ym, liquid, adv, exret_next, p_A ... p_OLS). Usage: python research/p1_paths.py [seed]"""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "."); from adr_longshort import X_COLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0; only_A = seed != 0
GRID = [(d, l) for d in (2, 4, 6) for l in (20, 50)]
M = pd.read_csv("research/out/p0_panel_corrected.csv"); M["ym"] = pd.PeriodIndex(M.ym, freq="M"); T = M[M.train_ok]
def wins(y): lo, hi = y.quantile([.01, .99]); return y.clip(lo, hi), (lo, hi)
def rf(d, l): return RandomForestRegressor(n_estimators=300, max_depth=d, min_samples_leaf=l, max_features=0.5, n_jobs=5, random_state=seed)
def fit_grid(tr, va, w=None):
    y, (lo, hi) = wins(tr.exret_next); yv = va.exret_next.clip(lo, hi).values; best = None
    for d, l in GRID:
        m = rf(d, l).fit(tr[X_COLS].values, y.values, sample_weight=w); e = float(((yv - m.predict(va[X_COLS].values)) ** 2).mean())
        if best is None or e < best[0]: best = (e, m, (d, l))
    return best[1], best[2]
def yrs(a, b): return T[(T.year >= a) & (T.year <= b)]
import os
out = []; t0 = time.time(); CK = 'research/out/p1_ck' + ('' if not only_A else f'_s{seed}'); os.makedirs(CK, exist_ok=True)
for Y in range(2008, 2027):
    if os.path.exists(f'{CK}/{Y}.csv'): out.append(pd.read_csv(f'{CK}/{Y}.csv').assign(ym=lambda d: pd.PeriodIndex(d.ym, freq='M'))); continue
    E = Y - 1; te = M[(M.year == Y) & M.score_ok].copy()
    if not len(te): continue
    X = te[X_COLS].values
    mA, hp = fit_grid(yrs(2001, E - 2), yrs(E - 1, E)); te["p_A"] = mA.predict(X)
    if not only_A:
        mB, _ = fit_grid(yrs(2001, E - 1), yrs(E, E)); te["p_B"] = mB.predict(X)
        trC = yrs(2001, E); yC, _ = wins(trC.exret_next); mC = rf(*hp).fit(trC[X_COLS].values, yC.values); te["p_C"] = mC.predict(X)
        for W, lab in [(6, "p_D6"), (8, "p_D8")]:
            m, _ = fit_grid(yrs(max(2001, E - 1 - W), E - 2), yrs(E - 1, E)); te[lab] = m.predict(X)
        trR = yrs(2001, E - 2); m, _ = fit_grid(trR, yrs(E - 1, E), w=0.5 ** ((E - 2 - trR.year.values) / 3)); te["p_R"] = m.predict(X)
        te["p_Q"] = te.p_C
        for qs in (4, 7, 10):                                            # refit at the start of Apr, Jul, Oct on every row before that month
            if not (te.ym.dt.month >= qs).any(): continue
            cut = pd.Period(f"{Y}-{qs:02d}", "M"); tr = T[(T.year >= 2001) & (T.ym < cut)]; y, _ = wins(tr.exret_next); m = rf(*hp).fit(tr[X_COLS].values, y.values)
            sel = te.ym.dt.month >= qs
            if not sel.any(): continue
            te.loc[sel, "p_Q"] = m.predict(te.loc[sel, X_COLS].values)
        trH = yrs(2001, E - 2); yH, _ = wins(trH.exret_next)
        te["p_HUB"] = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=2000).fit(trH[X_COLS].values, yH.values).predict(X)
        te["p_OLS"] = LinearRegression().fit(trH[X_COLS].values, yH.values).predict(X)
    te.to_csv(f'{CK}/{Y}.csv', index=False); out.append(te); print(f"{Y} hp {hp} [{time.time()-t0:.0f}s]", flush=True)
P = pd.concat(out); cols = [c for c in P.columns if c.startswith("p_")]
P = P[["ticker", "ym", "liquid", "adv", "exret_next", *cols]]
P.to_csv("research/out/p1_paths.csv" if not only_A else f"research/out/p1_paths_seed{seed}.csv", index=False)
