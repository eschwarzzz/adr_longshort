"""Phase 5 runs: China (shipped and corrected), Greater China extension, non-China foreign micro-listings, US microcaps, HKEX small caps.
    python research/p5_run.py
Native = refit on the universe itself with the repo protocol; transfer = the corrected China forests applied to the universe's own ranks."""
import sys, time, pickle, numpy as np, pandas as pd
sys.path.insert(0, "research"); sys.path.insert(0, ".")
from p5_pipeline import build, fit_predict, evaluate, bench_monthly, log_trial, D
from adr_longshort import load_prices, load_predictions
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); t0 = time.time()
rd = lambda n: pd.read_csv(f"{D}/{n}_prices.csv.gz", parse_dates=["date"]); rs = lambda n: pd.read_csv(f"{D}/{n}_splits.csv")
res, Q = [], {}
def run(label, F, bench, tid, extra=""):
    d, q, A = evaluate(F, bench, label=label); res.append(d); Q[label] = (q, A); q.to_csv(f"research/out/p5_monthly_{tid}.csv"); A.to_csv(f"research/out/p5_allq_{tid}.csv")
    log_trial(tid, f"{label}{extra}", f"rule LS {d.get('ls_sharpe', np.nan):.2f} net {d.get('net_sharpe', np.nan):.2f} ({d['months_rule']} m); all-tradeable LS {d['ls_all_sharpe']:.2f}; IC {d['rank_ic']:.3f}", f"research/out/p5_monthly_{tid}.csv")
    print(f"[{time.time()-t0:.0f}s] {label}: rule months {d['months_rule']}, LS {d.get('ls_sharpe', np.nan):.2f}, all-q LS {d['ls_all_sharpe']:.2f}, IC {d['rank_ic']:.3f}", flush=True)
PGJ = bench_monthly("PGJ")
# 0. China, shipped forecasts (no corrections), for reference
S0 = load_predictions(); S0["liquid"] = S0.adv >= 1e6; S0["tradeable"] = True; S0["year"] = S0.ym.dt.year
run("China, shipped (mean ADV, $5k rule, adjusted $1)", S0, PGJ, "p5_china_shipped")
# 1. China, corrected
PC = load_prices(); SC = rs("china_splits"); MC, st = build(PC, SC); MC.to_csv(f"{D}/panel_china_corrected.csv.gz", index=False)
FC, models = fit_predict(MC); pickle.dump(models, open(f"{D}/china_models.pkl", "wb")); FC.to_csv("research/out/p5_pred_china_corrected.csv", index=False)
run("China, corrected", FC, PGJ, "p5_china_corrected")
# 2. Greater China extension (adds Hong Kong / Macau-based and unlisted-in-USCC China names)
PG = pd.concat([PC, rd("gc_ext")]); SG = pd.concat([SC, rs("gc_ext")]); MG, _ = build(PG, SG); MG.to_csv(f"{D}/panel_gc.csv.gz", index=False)
FG, _ = fit_predict(MG); FG.to_csv("research/out/p5_pred_gc.csv", index=False); run("Greater China (China + HK/Macau ext.), native", FG, PGJ, "p5_gc_native")
ext = set(rd("gc_ext").ticker); run("HK/Macau ext. names only, scored by the Greater China model", FG[FG.ticker.isin(ext)], PGJ, "p5_gc_extonly")
# 3-5. other universes: native refit where history allows, and the China forests transferred
for name, label, fx, minp, bk in [("foreign", "Non-China Asia foreign listings (US exchanges)", 1.0, 1.0, "EEM"), ("us", "US microcaps (random 800, survivors)", 1.0, 1.0, "IWC"), ("hk", "HKEX small caps (random 1,000, HK$1 floor)", 7.8, 1.0, "2800.HK")]:
    M, _ = build(rd(name), rs(name), fx=fx, min_price=minp); M.to_csv(f"{D}/panel_{name}.csv.gz", index=False); B = bench_monthly(bk)
    Fn, _ = fit_predict(M, first_test=2010); Fn.to_csv(f"research/out/p5_pred_{name}_native.csv", index=False)
    if len(Fn): run(f"{label}, native", Fn, B, f"p5_{name}_native")
    Ft, _ = fit_predict(M, models=models); Ft.to_csv(f"research/out/p5_pred_{name}_transfer.csv", index=False); run(f"{label}, China forests", Ft, B, f"p5_{name}_transfer")
T = pd.DataFrame(res).set_index("universe"); T.to_csv("research/out/p5_summary.csv")
cols = ["first", "months_rule", "names_tradeable", "names_liquid", "n_short", "rank_ic", "ic_t", "ls_ret", "ls_sharpe", "net_sharpe", "short_alone_sharpe", "long_ew_sharpe", "beta_ls", "beta_short_leg", "ls_sh_10_15", "ls_sh_16_20", "ls_sh_21_26", "months_all", "ls_all_sharpe", "all_sh_10_15", "all_sh_16_20", "all_sh_21_26", "q1_all_minus_ew"]
print(T[[c for c in cols if c in T]].round(2).T.to_string()); print(f"done [{time.time()-t0:.0f}s]")
