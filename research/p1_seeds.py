"""Phase 1.3 seeds: RF shipped-freeze path for random_state 0-4 and the RF+OLS rank blend per seed; unit and account metrics,
mean and spread across seeds, paired t of RF minus Huber / OLS per seed. Writes research/out/p1_seeds.csv."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "research")
from adr_longshort import load_predictions
from metrics import full_report, unit_book
from trials import log
B = load_predictions("research/out/p1_paths_2010.csv")
qH = unit_book(B, "p_HUB"); qO = unit_book(B, "p_OLS"); rows = []
for s in range(5):
    if s == 0: P = B[["ticker", "ym", "liquid", "adv", "exret_next", "p_A"]].copy()
    else: P = load_predictions(f"research/out/p1_paths_seed{s}.csv"); P = P[P.ym.dt.year >= 2010].copy()
    P = P.merge(B[["ticker", "ym", "p_OLS"]], on=["ticker", "ym"])
    P["p_BL_RO"] = (P.groupby("ym").p_A.rank(pct=True) + P.groupby("ym").p_OLS.rank(pct=True)) / 2
    f = f"research/out/p1_seed{s}_2010.csv"; P.to_csv(f, index=False)
    for c, lab in [("p_A", "RF"), ("p_BL_RO", "blend RF+OLS")]:
        o, q = full_report(f, c); dH = (q.LS_net - qH.LS_net.reindex(q.index)).dropna(); dO = (q.LS_net - qO.LS_net.reindex(q.index)).dropna()
        rows.append(dict(seed=s, model=lab, unit_sharpe=o["unit_sharpe"], unit_net_sharpe=o["unit_net_sharpe"], short_ret=o["short_ret"], acct_sharpe=o["acct_sharpe"], acct_cagr=o["acct_cagr_pct"], acct_maxdd=o["acct_maxdd_pct"],
                         t_minus_huber=dH.mean() / dH.std() * np.sqrt(len(dH)), t_minus_ols=dO.mean() / dO.std() * np.sqrt(len(dO))))
        log("P1", f"1.3 seed {s} {lab}", f"unit {o['unit_sharpe']:.2f} acct {o['acct_sharpe']:.2f}", f)
S = pd.DataFrame(rows); S.to_csv("research/out/p1_seeds.csv", index=False)
print(S.round(2).to_string(index=False)); print(S.groupby("model").agg(["mean", "std"]).drop(columns="seed").round(2).T.to_string())
