"""Greater China extension rerun: the screener leaves HKD and AMTD with a blank country, so they are added by hand; CPRI (Capri Holdings, not
Hong Kong-based) is dropped. Replaces the two Greater China rows of research/out/p5_summary.csv."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "research"); sys.path.insert(0, ".")
from p5_pipeline import build, fit_predict, evaluate, bench_monthly, log_trial, D
from adr_longshort import load_prices
rd = lambda n: pd.read_csv(f"{D}/{n}_prices.csv.gz", parse_dates=["date"]); rs = lambda n: pd.read_csv(f"{D}/{n}_splits.csv")
E = pd.concat([rd("gc_ext"), rd("gc_add")]); E = E[E.ticker != "CPRI"]; ES = pd.concat([rs("gc_ext"), rs("gc_add")]); ES = ES[ES.ticker != "CPRI"]
PG = pd.concat([load_prices(), E]); SG = pd.concat([rs("china_splits"), ES]); MG, _ = build(PG, SG); MG.to_csv(f"{D}/panel_gc.csv.gz", index=False)
FG, _ = fit_predict(MG); FG.to_csv("research/out/p5_pred_gc.csv", index=False); PGJ = bench_monthly("PGJ"); T = pd.read_csv("research/out/p5_summary.csv").set_index("universe")
for label, F, tid in [("Greater China (China + HK/Macau ext.), native", FG, "p5_gc_native"), ("HK/Macau ext. names only, scored by the Greater China model", FG[FG.ticker.isin(set(E.ticker))], "p5_gc_extonly")]:
    d, q, A = evaluate(F, PGJ, label=label); q.to_csv(f"research/out/p5_monthly_{tid}.csv"); A.to_csv(f"research/out/p5_allq_{tid}.csv"); T.loc[label] = pd.Series(d).drop("universe")
    log_trial(tid + "_v2", label + " (HKD, AMTD added; CPRI dropped)", f"rule LS {d.get('ls_sharpe', np.nan):.2f} net {d.get('net_sharpe', np.nan):.2f}; all-tradeable LS {d['ls_all_sharpe']:.2f}; IC {d['rank_ic']:.3f}", f"research/out/p5_monthly_{tid}.csv")
    print(label, {k: round(v, 2) if isinstance(v, float) else v for k, v in d.items() if k in ("months_rule", "ls_sharpe", "net_sharpe", "ls_all_sharpe", "rank_ic", "ls_sh_21_26", "names_tradeable")})
T.to_csv("research/out/p5_summary.csv")
