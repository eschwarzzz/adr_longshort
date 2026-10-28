"""Phase 2.3 (zero_1m inside the protocol) and 2.4(b) (redundant five), hyperparameters re-chosen on validation MSE each year (seed 0);
seeds 1-4 reuse each year's seed-0 choice. 2.3 selection rule, pre-specified: each December keep zero_1m iff the 16-feature forest's
validation long-short spread (liquid names, $3m short floor, mean monthly) beats the 15-feature forest's; secondary rule: validation rank IC."""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "research"); from p2_common import fit_variant, write_preds, report, KEEP, panel
from adr_longshort import FEATS
from trials import log
t0 = time.time(); pd.set_option("display.width", 300)
ALL = list(FEATS); FIVE = ["beta_12m", "minret_1m", "illiq_1m", "maxret_1m", "age"]
SETS = {"all16": ALL, "drop zero_1m": [f for f in ALL if f != "zero_1m"]}
for f in FIVE: SETS[f"drop {f}"] = [g for g in ALL if g != f]
SETS["drop five"] = [g for g in ALL if g not in FIVE]; SETS["drop five+zero_1m"] = [g for g in ALL if g not in FIVE + ["zero_1m"]]
preds = {}; infos = {}
for s in range(5):
    for name, fs in SETS.items():
        if s == 0: p, i = fit_variant(fs, hp="grid", seed=0, want_val=True); infos[name] = i
        else: p, i = fit_variant(fs, seed=s, hyper={Y: (int(r.depth), int(r.leaf)) for Y, r in infos[name].iterrows()}, want_val=name in ("all16", "drop zero_1m")); infos[(name, s)] = i
        preds[f"{name}|s{s}"] = p
        if name in ("all16", "drop zero_1m"): infos[(name, s)] = i
    # selected path for this seed
    a, b = infos[("all16", s)], infos[("drop zero_1m", s)]
    keep = a.val_ls >= b.val_ls; keep_ic = a.val_ic >= b.val_ic
    M, _ = panel(); pa, pb = preds[f"all16|s{s}"], preds[f"drop zero_1m|s{s}"]; yr = M.loc[pa.index, "year"]
    preds[f"selected(val LS)|s{s}"] = pd.Series(np.where(keep.reindex(yr).values, pa.values, pb.reindex(pa.index).values), index=pa.index)
    preds[f"selected(val IC)|s{s}"] = pd.Series(np.where(keep_ic.reindex(yr).values, pa.values, pb.reindex(pa.index).values), index=pa.index)
    if s == 0:
        sel = pd.DataFrame({"val_ls_16": a.val_ls, "val_ls_15": b.val_ls, "keep_zero_by_LS": keep, "val_ic_16": a.val_ic, "val_ic_15": b.val_ic, "keep_zero_by_IC": keep_ic, "depth16": a.depth, "leaf16": a.leaf, "depth15": b.depth, "leaf15": b.leaf})
        sel.to_csv("research/out/p2_zero_selection_by_year.csv"); print(sel.round(4).to_string(), flush=True)
    print("seed", s, f"[{time.time()-t0:.0f}s]", flush=True)
path = write_preds(preds, "research/out/p2_pred_confirm.csv")
R = report(path, list(preds)); R.to_csv("research/out/p2_confirm_all_seeds.csv")
R["set"] = [k.split("|")[0] for k in R.index]; num = R.select_dtypes("number")
G = num.groupby(R["set"]).agg(["mean", "std"]); G.to_csv("research/out/p2_confirm_seed_summary.csv")
S0 = R[[k.endswith("|s0") for k in R.index]]
for k, r in R.iterrows(): log("P2", f"2.3/2.4 {k} (grid hp)", f"unit {r.unit_sharpe:.2f}, net {r.unit_net_sharpe:.2f}, acct {r.acct_sharpe:.2f}, t {r.paired_t_vs_base:.2f}", "research/out/p2_confirm_all_seeds.csv")
print(S0[KEEP].round(2).to_string()); print(G[[(c, s) for c in ["unit_sharpe", "unit_net_sharpe", "acct_sharpe", "acct_maxdd_pct", "paired_t_vs_base"] for s in ("mean", "std")]].round(2).to_string())
print(f"done [{time.time()-t0:.0f}s]")
