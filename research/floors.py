"""Quintile means over the liquid universe and long-short Sharpe by short floor, per forecaster."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "."); from adr_longshort import load_predictions, holdings
F = load_predictions("research/out/predictions_alt.csv"); L = F[F.liquid]
sr = lambda x: x.mean() / x.std() * np.sqrt(12)
rows = []
for c in ["p_RF", "p_HUB", "p_OLS", "p_SIGN", "p_COMBO3", "p_RFTV"]:
    q = L.groupby("ym").filter(lambda g: len(g) >= 25).copy(); q["q"] = q.groupby("ym")[c].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    qm = q.groupby(["ym", "q"]).exret_next.mean().unstack(); d = dict(model=c, **{f"Q{k}": qm[k].mean() * 1200 for k in range(1, 6)})
    for fl in [0, 1e6, 3e6, 1e7, 2.5e7]:
        ls = [];
        for ym, g in L.groupby("ym"):
            h = holdings(g, c, short_floor=fl)
            if h: ls.append(g[g.ticker.isin(h["long"])].exret_next.mean() - g[g.ticker.isin(h["short"])].exret_next.mean())
        d[f"LS_floor_{fl/1e6:g}m"] = sr(pd.Series(ls))
    # deciles: where does the predictive power sit? bottom-decile vs rest
    q["d"] = q.groupby("ym")[c].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1); dm = q.groupby(["ym", "d"]).exret_next.mean().unstack().mean() * 1200
    d["D1"] = dm[1]; d["D10"] = dm[10]; rows.append(d)
print(pd.DataFrame(rows).set_index("model").round(2).to_string())
