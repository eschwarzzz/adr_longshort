"""Phase 2.1 build-up path and add-one table; 2.2 pairs. Hyperparameters fixed to the corrected baseline's per-year choices (exploratory)."""
import sys, time, itertools, numpy as np, pandas as pd
sys.path.insert(0, "research"); from p2_common import fit_variant, write_preds, report, KEEP
from trials import log
t0 = time.time(); pd.set_option("display.width", 300)
BASE = ["high52", "mom_6_1", "mom_12_1", "mom_36_13"]; ORDER = ["ret_1m", "vol_1m", "vol_3m", "ivol_12m", "beta_12m", "maxret_1m", "minret_1m", "dolvol_1m", "dolvol_trend", "illiq_1m", "age"]
part = sys.argv[1] if len(sys.argv) > 1 else "all"
if part in ("all", "buildup"):
    P = {}; fs = list(BASE); P["base"] = fit_variant(fs)[0]; fs = fs + ["zero_1m"]; P["+zero_1m"] = fit_variant(fs)[0]
    for f in ORDER: fs = fs + [f]; P[f"+{f}"] = fit_variant(fs)[0]; print("path", f, f"[{time.time()-t0:.0f}s]", flush=True)
    for f in ["zero_1m"] + ORDER: P[f"base+{f} (single)"] = fit_variant(BASE + [f])[0]
    path = write_preds(P, "research/out/p2_pred_buildup.csv"); R = report(path, list(P)); R.to_csv("research/out/p2_buildup.csv")
    for k, r in R.iterrows(): log("P2", f"2.1 feature set {k} (fixed hp)", f"unit {r.unit_sharpe:.2f}, net {r.unit_net_sharpe:.2f}, acct {r.acct_sharpe:.2f}, t {r.paired_t_vs_base:.2f}", "research/out/p2_buildup.csv")
    print(R[KEEP].round(2).to_string(), flush=True)
if part in ("all", "pairs"):
    EIGHT = ["high52", "ivol_12m", "mom_6_1", "dolvol_trend", "mom_12_1", "vol_1m", "dolvol_1m", "zero_1m"]; P = {}
    for f in EIGHT: P[f"single {f}"] = fit_variant([f])[0]
    for a, b in itertools.combinations(EIGHT, 2):
        P[f"pair {a}+{b}"] = fit_variant([a, b])[0]
        P[f"base+{a}+{b}"] = fit_variant(list(dict.fromkeys(BASE + [a, b])))[0]
    print("pairs fitted", f"[{time.time()-t0:.0f}s]", flush=True)
    path = write_preds(P, "research/out/p2_pred_pairs.csv"); R = report(path, list(P), account=False); R.to_csv("research/out/p2_pairs.csv")
    for k, r in R.iterrows(): log("P2", f"2.2 {k} (fixed hp)", f"unit {r.unit_sharpe:.2f}, net {r.unit_net_sharpe:.2f}, t {r.paired_t_vs_base:.2f}", "research/out/p2_pairs.csv")
    print("done pairs", f"[{time.time()-t0:.0f}s]")
