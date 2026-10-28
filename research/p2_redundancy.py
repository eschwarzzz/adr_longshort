"""Phase 2.4(a): are the 'redundant five' plausibly redundant? Rank correlations with the other features and R^2 of each rank on the other 15,
on the corrected panel's scored rows (all tradeable) and on the liquid subset."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "research"); from p2_common import panel
from adr_longshort import FEATS
from sklearn.linear_model import LinearRegression
M, _ = panel(); S = M[M.score_ok & (M.year >= 2010)]; FIVE = ["beta_12m", "minret_1m", "illiq_1m", "maxret_1m", "age"]; rows = []
for name, D in [("all tradeable", S), ("liquid", S[S.liquid])]:
    X = D[[f + "_r" for f in FEATS]]; C = X.corr()
    for f in FIVE + ["zero_1m"]:
        others = [g + "_r" for g in FEATS if g != f]; r2 = LinearRegression().fit(X[others], X[f + "_r"]).score(X[others], X[f + "_r"])
        top = C[f + "_r"].drop(f + "_r").abs().sort_values(ascending=False).head(3)
        rows.append(dict(universe=name, feature=f, r2_on_other15=r2, top_corr=", ".join(f"{k[:-2]} {C.loc[f+'_r', k]:+.2f}" for k in top.index)))
T = pd.DataFrame(rows); T.to_csv("research/out/p2_redundancy_spanning.csv", index=False); pd.set_option("display.width", 250); print(T.round(3).to_string(index=False))
