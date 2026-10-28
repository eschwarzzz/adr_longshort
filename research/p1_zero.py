"""Phase 1.1: where does zero_1m (and every other feature) carry information? Rank IC and quintile spreads of each raw feature
within the >=$100k (scored), >=$1m and >=$3m sets (median ADV), 2010-2026; zero_1m distribution by set; forest importances when
fitted on the full panel vs only >=$1m rows. Writes research/out/p1_zero_ic.csv, p1_zero_dist.csv, p1_zero_importance.csv."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "research")
from adr_longshort import FEATS, X_COLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from trials import log
M = pd.read_csv("research/out/p0_panel_corrected.csv"); M["ym"] = pd.PeriodIndex(M.ym, freq="M")
E = M[M.score_ok & M.exret_next.notna() & (M.year >= 2010)].copy()
SETS = {">=$100k (scored)": E, ">=$1m": E[E.adv >= 1e6], ">=$3m": E[E.adv >= 3e6]}
rows = []
for sname, S in SETS.items():
    for f in FEATS:
        x = S.dropna(subset=[f])
        ic = x.groupby("ym").apply(lambda g: g[f].rank().corr(g.exret_next.rank()) if len(g) >= 10 else np.nan, include_groups=False).dropna()
        def qs(g):
            if len(g) < 10: return pd.Series(dtype=float)
            q = pd.qcut(g[f].rank(method="first"), 5, labels=False) + 1; return g.groupby(q).exret_next.mean()
        Q = x.groupby("ym").apply(qs, include_groups=False)
        Q = Q.unstack() if isinstance(Q, pd.Series) else Q
        sp = (Q[5] - Q[1]).dropna()
        rows.append(dict(set=sname, feature=f, months=len(ic), names_per_month=x.groupby("ym").size().mean(), ic=ic.mean(), ic_t=ic.mean() / ic.std() * np.sqrt(len(ic)),
                         **{f"Q{k}": Q[k].mean() * 1200 for k in range(1, 6)}, q5_q1=sp.mean() * 1200, q5_q1_t=sp.mean() / sp.std() * np.sqrt(len(sp))))
IC = pd.DataFrame(rows); IC.to_csv("research/out/p1_zero_ic.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
print(IC.pivot(index="feature", columns="set", values="ic").round(3).join(IC.pivot(index="feature", columns="set", values="ic_t").round(1), rsuffix=" t").to_string())
print("\nzero_1m quintiles by set (% a year):"); print(IC[IC.feature == "zero_1m"].set_index("set")[["names_per_month", "Q1", "Q2", "Q3", "Q4", "Q5", "q5_q1", "q5_q1_t"]].round(2).to_string())
D = []
for sname, S in SETS.items():
    z = S.zero_1m; D.append(dict(set=sname, rows=len(z), share_exactly_zero=(z == 0).mean(), share_gt_10pct=(z > 0.10).mean(), share_gt_25pct=(z > 0.25).mean(), **{f"p{p}": z.quantile(p / 100) for p in (10, 25, 50, 75, 90, 95, 99)}))
D = pd.DataFrame(D).set_index("set"); D.to_csv("research/out/p1_zero_dist.csv"); print("\nzero_1m distribution:"); print(D.round(3).to_string())
# forest importances: full panel vs >=$1m only (train window of the 2026 model: 2001-2023), two tree settings
T = M[M.train_ok & (M.year >= 2001) & (M.year <= 2023)]; V = M[M.train_ok & (M.year >= 2024) & (M.year <= 2025)]
imp = {}
for lab, tr in [("full", T), ("liquid_only", T[T.adv >= 1e6])]:
    lo, hi = tr.exret_next.quantile([.01, .99]); y = tr.exret_next.clip(lo, hi)
    for d, l in [(2, 20), (6, 20)]:
        rf = RandomForestRegressor(n_estimators=300, max_depth=d, min_samples_leaf=l, max_features=0.5, n_jobs=4, random_state=0).fit(tr[X_COLS].values, y.values)
        imp[f"{lab}_d{d}_mdi"] = rf.feature_importances_
        vl = V[V.adv >= 1e6]; pi = permutation_importance(rf, vl[X_COLS].values, vl.exret_next.clip(lo, hi).values, n_repeats=5, random_state=0, n_jobs=4)
        imp[f"{lab}_d{d}_perm_on_liquid_val"] = pi.importances_mean * 1e4
I = pd.DataFrame(imp, index=FEATS); I["rank_full_d2"] = I.full_d2_mdi.rank(ascending=False); I["rank_liq_d2"] = I.liquid_only_d2_mdi.rank(ascending=False)
I["rank_full_d6"] = I.full_d6_mdi.rank(ascending=False); I["rank_liq_d6"] = I.liquid_only_d6_mdi.rank(ascending=False)
I.to_csv("research/out/p1_zero_importance.csv"); print("\nforest importances, train 2001-2023 (perm in 1e-4 MSE units on 2024-25 liquid rows):"); print(I.sort_values("full_d6_mdi", ascending=False).round(3).to_string())
log("P1", "1.1 feature IC/quintiles by liquidity set; zero_1m distribution; forest importance full vs liquid-only", f"zero_1m IC: " + ", ".join(f"{s}={v:.3f}" for s, v in IC[IC.feature == "zero_1m"].set_index("set").ic.items()), "research/out/p1_zero_ic.csv")
