"""Phase 1.2/1.3 selection and reporting. Inputs: research/out/p1_paths.csv (+ p1_paths_seed{1..4}.csv for RF seeds).
Selection inside the protocol, PRE-SPECIFIED: at each December Y-1, every candidate's own out-of-sample path over the 24 months whose
targets are realised by then (rows Dec Y-3 .. Nov Y-1) is scored by its mean monthly unit long-short (liquid set, $3m short floor,
adr_longshort.holdings, gross); the best is used for year Y. Secondary: trailing 24-month mean rank IC. No candidate trained on the rows
it is judged on (each path row comes from a model fitted on data ending the year before)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "research")
from adr_longshort import load_predictions, holdings
from metrics import full_report, unit_book
from trials import log
P = load_predictions("research/out/p1_paths.csv")
def xrank(c): return P.groupby("ym")[c].rank(pct=True)
P["p_BL_RH"] = (xrank("p_A") + xrank("p_HUB")) / 2; P["p_BL_RO"] = (xrank("p_A") + xrank("p_OLS")) / 2
REC = ["p_A", "p_B", "p_C", "p_D6", "p_D8", "p_R", "p_Q"]; MOD = ["p_A", "p_HUB", "p_OLS", "p_BL_RH", "p_BL_RO"]
U = {c: unit_book(P, c) for c in sorted(set(REC + MOD))}
IC = {c: P.dropna(subset=["exret_next"]).groupby("ym").apply(lambda g: g[c].rank().corr(g.exret_next.rank()), include_groups=False) for c in U}
def select(cands, use_ic=False):
    choice = {}
    for Y in range(2010, 2027):
        lo, hi = pd.Period(f"{Y-3}-12", "M"), pd.Period(f"{Y-1}-11", "M")
        sc = {c: (IC[c] if use_ic else U[c].LS).loc[lo:hi].mean() for c in cands}; choice[Y] = max(sc, key=sc.get)
    return choice
sels = {"p_SEL_REC": select(REC), "p_SEL_REC_IC": select(REC, True), "p_SEL_MOD": select(MOD), "p_SEL_MOD_IC": select(MOD, True)}
for name, ch in sels.items(): P[name] = [r[ch[r.ym.year]] if r.ym.year in ch else np.nan for _, r in P[[*set(ch.values()), "ym"]].assign(ym=P.ym).iterrows()]
pd.DataFrame(sels).to_csv("research/out/p1_selection_choices.csv"); print("choices by test year:"); print(pd.DataFrame(sels).to_string())
P2 = P[P.ym.dt.year >= 2010].copy(); P2.to_csv("research/out/p1_paths_2010.csv", index=False)
qA = unit_book(P2, "p_A")
def rep(cols, tag, labels):
    rows = []
    for c in cols:
        o, q = full_report("research/out/p1_paths_2010.csv", c); d = (q.LS_net - qA.LS_net.reindex(q.index)).dropna()
        o["paired_t_vs_A"] = d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else 0.0
        icc = IC.get(c, P2.dropna(subset=["exret_next"]).groupby("ym").apply(lambda g: g[c].rank().corr(g.exret_next.rank()), include_groups=False)).loc["2010-01":]
        o["rank_ic"] = icc.mean(); o["rank_ic_t"] = icc.mean() / icc.std() * np.sqrt(len(icc))
        rows.append(dict(variant=labels.get(c, c), col=c, **o)); log("P1", f"{tag}: {labels.get(c, c)}", f"unit {o['unit_sharpe']:.2f}, net {o['unit_net_sharpe']:.2f}, acct {o['acct_sharpe']:.2f}, t_vs_A {o['paired_t_vs_A']:.2f}", f"research/out/p1_{tag}.csv")
    T = pd.DataFrame(rows).set_index("variant"); T.to_csv(f"research/out/p1_{tag}.csv"); return T
L = {"p_A": "A shipped freeze", "p_B": "B 1-yr validation", "p_C": "C retrain through Y-1", "p_D6": "D6 rolling 6y", "p_D8": "D8 rolling 8y", "p_R": "R recency-weighted", "p_Q": "Q quarterly refit",
     "p_SEL_REC": "selected by trailing LS", "p_SEL_REC_IC": "selected by trailing IC", "p_HUB": "Huber", "p_OLS": "OLS", "p_BL_RH": "blend RF+Huber", "p_BL_RO": "blend RF+OLS",
     "p_SEL_MOD": "model selected by trailing LS", "p_SEL_MOD_IC": "model selected by trailing IC"}
K = ["rank_ic", "unit_ret", "unit_sharpe", "unit_net_sharpe", "unit_net_harsh", "long_ret", "short_ret", "beta_pgj", "unit_maxdd", "unit_sharpe_2010_15", "unit_sharpe_2016_20", "unit_sharpe_2021_26", "paired_t_vs_A", "paired_t_vs_base",
     "acct_final_equity", "acct_cagr_pct", "acct_sharpe", "acct_maxdd_pct", "acct_worst_month_pct", "acct_beta_pgj", "acct_sharpe_2021_26"]
pd.set_option("display.width", 300)
R = rep(REC + ["p_SEL_REC", "p_SEL_REC_IC"], "recency", L); print("\n1.2 refit recency:"); print(R[K].round(2).to_string())
Mo = rep(["p_HUB", "p_OLS", "p_BL_RH", "p_BL_RO", "p_SEL_MOD", "p_SEL_MOD_IC"], "models", L); Mo = pd.concat([R.loc[["A shipped freeze"]], Mo]); Mo.to_csv("research/out/p1_models.csv")
print("\n1.3 models:"); print(Mo[K].round(2).to_string())
# 1.3(b): where the money is: quintiles within the >=$1m long set and the >=$3m short set separately
rows = []
for c in ["p_A", "p_HUB", "p_OLS", "p_BL_RH", "p_BL_RO"]:
    for nm, fl in [(">=$1m", 1e6), (">=$3m", 3e6)]:
        S = P2[(P2.adv >= fl) & P2.exret_next.notna()].copy()
        S["q"] = S.groupby("ym")[c].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1 if len(s) >= 10 else np.nan)
        Q = S.groupby(["ym", "q"]).exret_next.mean().unstack(); sp = Q[5.0] - Q[1.0]
        rows.append(dict(model=L[c], set=nm, Q1=Q[1.0].mean() * 1200, Q5=Q[5.0].mean() * 1200, q5_q1=sp.mean() * 1200, q5_q1_t=sp.mean() / sp.std() * np.sqrt(len(sp.dropna())), q1_t=Q[1.0].mean() / Q[1.0].std() * np.sqrt(len(Q))))
QQ = pd.DataFrame(rows); QQ.to_csv("research/out/p1_models_quintiles.csv", index=False); print("\n1.3(b) quintile spreads inside the traded sets (% a year):"); print(QQ.round(2).to_string(index=False))
# 1.3(c): paired t of monthly unit NET returns RF minus each alternative
rows = []
for c in ["p_HUB", "p_OLS", "p_BL_RH", "p_BL_RO"]:
    q = unit_book(P2, c); d = (qA.LS_net - q.LS_net.reindex(qA.index)).dropna(); rows.append(dict(comparison=f"RF minus {L[c]}", months=len(d), mean_diff_pct_yr=d.mean() * 1200, t=d.mean() / d.std() * np.sqrt(len(d)), corr=qA.LS_net.corr(q.LS_net)))
PT = pd.DataFrame(rows); PT.to_csv("research/out/p1_models_paired.csv", index=False); print("\n1.3(c) paired tests (unit net):"); print(PT.round(2).to_string(index=False))
