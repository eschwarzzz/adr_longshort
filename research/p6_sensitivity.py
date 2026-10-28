"""Sensitivity grids on the $500k account: sizing, costs, borrow, cash interest. Writes research/out/p6_sens_*.csv."""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "research"); from account import load
t0 = time.time(); B = load("research/out/p0_pred_corrected_reuse.csv"); print(f"loaded [{time.time()-t0:.0f}s]", flush=True)
KEEP = ["final_equity", "cagr_pct", "vol_pct", "sharpe", "sharpe_monthly", "maxdd_pct", "worst_month_pct", "avg_side_pct", "peak_margin_pct", "fees_usd", "borrow_usd", "interest_usd"]
def run(label, **kw):
    d, m = B.simulate(**kw); s = B.stats(d, m); s["sharpe_ex_cash"] = ((d.pnl - d.interest) / d.equity.shift(1).fillna(5e5)).pipe(lambda r: r.mean() / r.std() * np.sqrt(252))
    s["sharpe_2021_26"] = d.ret.loc["2021":].pipe(lambda r: r.mean() / r.std() * np.sqrt(252)); return dict(variant=label, **{k: s[k] for k in KEEP + ["sharpe_ex_cash", "sharpe_2021_26"]})
pd.set_option("display.width", 250)
def show(rows, name):
    T = pd.DataFrame(rows).set_index("variant"); T.to_csv(f"research/out/p6_sens_{name}.csv"); print(f"\n== {name} [{time.time()-t0:.0f}s]"); print(T.round(2).to_string(), flush=True)
# 1. sizing: vol target x cap, fixed exposure
show([run(f"target {t:.0%}, cap {c:.0%}", target=t, cap=c) for t in (0.05, 0.10, 0.15, 0.20) for c in (0.5, 0.8)] + [run(f"fixed {f:.0%} a side", fixed=f) for f in (0.15, 0.25, 0.35, 0.50)], "sizing")
# 2. vol estimator: window and blend weight
rows = []
for w in (20, 60, 120, 250):
    for a in (1.0, 0.7, 0.4):
        v = B.U.rolling(w, min_periods=min(20, w)).std() * np.sqrt(252); vol = a * v + (1 - a) * v.expanding(20).mean(); rows.append(run(f"window {w}d, weight on recent {a:.1f}", vol=vol))
rows.append(run("expanding mean only (no recent)", vol=(B.U.rolling(60, min_periods=20).std() * np.sqrt(252)).expanding(20).mean()))
show(rows, "volestimator")
# 3. costs per side and borrow schedules
B.BORROW.update({"base_x2": [(lo, 2 * r) for lo, r in B.BORROW["base"]], "base_x3": [(lo, 3 * r) for lo, r in B.BORROW["base"]], "flat15": [(0, .15)], "flat25": [(0, .25)], "flat40": [(0, .40)]})
show([run(f"cost {c*1e4:.0f} bp/side", cost=c) for c in (0.0005, 0.001, 0.002, 0.003, 0.004, 0.006, 0.010)], "costs")
show([run(f"borrow {b}", borrow=b) for b in ("zero", "flat3", "base", "harsh", "base_x2", "base_x3", "flat15", "flat25", "flat40")], "borrow")
show([run(f"cost {c*1e4:.0f}bp + borrow {b}", cost=c, borrow=b) for c in (0.002, 0.004) for b in ("base", "harsh", "base_x3")], "costs_x_borrow")
