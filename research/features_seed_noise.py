"""Seed noise for the ablation: baseline (16) and drop zero_1m / drop liquidity family with random_state 0..4."""
import sys, numpy as np, pandas as pd
sys.argv = [sys.argv[0], "none"]          # load helpers from features_ablation without running stages
sys.path.insert(0, "research"); import features_ablation as fa
from sklearn.ensemble import RandomForestRegressor
def fp(feats, seed):
    cols = [f + "_r" for f in feats]; p = pd.Series(np.nan, index=fa.M.index)
    for Y in fa.YEARS:
        tr, va, te = fa.split(Y); m = RandomForestRegressor(n_estimators=300, max_depth=int(fa.HP.loc[Y, "max_depth"]), min_samples_leaf=int(fa.HP.loc[Y, "min_samples_leaf"]), max_features=0.5, n_jobs=-1, random_state=seed)
        p.loc[te.index] = m.fit(tr[cols].values, fa.wins(tr.exret_next, tr.exret_next).values).predict(te[cols].values)
    return p
V = {"baseline (16)": fa.FEATS, "drop zero_1m": [f for f in fa.FEATS if f != "zero_1m"], "drop liquidity family": [f for f in fa.FEATS if f not in fa.FAM["liquidity"]], "drop ivol_12m": [f for f in fa.FEATS if f != "ivol_12m"]}
rows = []
for name, fs in V.items():
    for s in range(5):
        r, _ = fa.evaluate(fp(fs, s), name); rows.append(dict(r, seed=s)); print(name, s, round(r["ic"], 3), round(r["ls_sharpe"], 2), flush=True)
D = pd.DataFrame(rows); D.to_csv("research/out/ablation_seed_noise.csv", index=False)
print(D.groupby("variant")[["ic", "ls_sharpe", "net_sharpe", "sh_10_15", "sh_16_20", "sh_21_26"]].agg(["mean", "std"]).round(3).to_string())
