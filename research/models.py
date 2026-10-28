"""Alternative forecasters through the repo's protocol (train 2001..Y-3, validate Y-2..Y-1 where there is a choice, score year Y).
Writes research/out/predictions_alt.csv: ticker, ym, liquid, adv, exret_next, p_RF (refit, should match shipped), p_HUB, p_OLS, p_SIGN, p_H52, p_COMBO3, p_RF4.
Usage: python research/models.py [panel.csv]"""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "."); from adr_longshort import annual_refit, predict, rank_transform, FEATS, X_COLS
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
path = sys.argv[1] if len(sys.argv) > 1 else "results/panel.csv"; tag = sys.argv[2] if len(sys.argv) > 2 else "alt"
M = pd.read_csv(path); M["ym"] = pd.PeriodIndex(M.ym, freq="M"); M = M[M.in_sample].copy(); M = rank_transform(M); M["liquid"] = M.adv >= 1e6
F4 = ["zero_1m_r", "high52_r", "ivol_12m_r", "mom_6_1_r"]
def wins(tr): lo, hi = tr.exret_next.quantile([.01, .99]); return tr.exret_next.clip(lo, hi)
out = []; t0 = time.time(); hub_coefs = {}
for Y in range(2010, 2027):
    te = M[M.year == Y].copy()
    if not len(te): continue
    tr = M[(M.year >= 2001) & (M.year <= Y - 3)]; va = M[(M.year >= Y - 2) & (M.year <= Y - 1)]; trva = M[(M.year >= 2001) & (M.year <= Y - 1)]
    y = wins(tr)
    m, info = annual_refit(M, Y); te["p_RF"] = predict(m, te)
    hub = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=2000).fit(tr[X_COLS].values, y.values); te["p_HUB"] = hub.predict(te[X_COLS].values); hub_coefs[Y] = hub.coef_
    te["p_OLS"] = LinearRegression().fit(tr[X_COLS].values, y.values).predict(te[X_COLS].values)
    # sign-only equal-weight composite: sign of each feature's pooled rank IC on train, then sum of signed ranks (no magnitudes fitted)
    sg = np.array([np.sign(spearmanr(tr[c], tr.exret_next).statistic) for c in X_COLS]); te["p_SIGN"] = te[X_COLS].values @ sg
    te["p_H52"] = te.high52_r; te["p_COMBO3"] = te.high52_r - te.zero_1m_r - te.ivol_12m_r          # fixed intuitive composite, no fitting
    rf4 = RandomForestRegressor(n_estimators=300, max_depth=info["max_depth"], min_samples_leaf=info["min_samples_leaf"], max_features=0.5, n_jobs=-1, random_state=0).fit(tr[F4].values, y.values)
    te["p_RF4"] = rf4.predict(te[F4].values)
    # RF trained on train+validation with the chosen hyperparameters (uses the two most recent years the protocol leaves out)
    y2 = wins(trva); te["p_RFTV"] = RandomForestRegressor(n_estimators=300, max_depth=info["max_depth"], min_samples_leaf=info["min_samples_leaf"], max_features=0.5, n_jobs=-1, random_state=0).fit(trva[X_COLS].values, y2.values).predict(te[X_COLS].values)
    out.append(te); print(f"{Y} done [{time.time()-t0:.0f}s]", flush=True)
P = pd.concat(out)
cols = ["p_RF", "p_HUB", "p_OLS", "p_SIGN", "p_H52", "p_COMBO3", "p_RF4", "p_RFTV"]
P[["ticker", "ym", "liquid", "adv", "exret_next", *cols]].to_csv(f"research/out/predictions_{tag}.csv", index=False)
pd.DataFrame(hub_coefs, index=FEATS).round(4).to_csv(f"research/out/huber_coefs_{tag}.csv")
ev = []
for c in cols:
    ic = P.groupby("ym").apply(lambda g: spearmanr(g[c], g.exret_next).statistic, include_groups=False); icl = P[P.liquid].groupby("ym").apply(lambda g: spearmanr(g[c], g.exret_next).statistic, include_groups=False)
    r2 = 1 - ((P.exret_next - P[c]) ** 2).sum() / (P.exret_next ** 2).sum() if c in ("p_RF", "p_HUB", "p_OLS", "p_RF4", "p_RFTV") else np.nan
    ev.append(dict(model=c, r2_pct=r2 * 100, ic=ic.mean(), ic_t=ic.mean() / ic.std() * np.sqrt(len(ic)), ic_liq=icl.mean(), ic_liq_t=icl.mean() / icl.std() * np.sqrt(len(icl))))
print(pd.DataFrame(ev).set_index("model").round(3).to_string())
