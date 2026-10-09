"""Daily prices -> the monthly research panel (results/panel.csv): 16 characteristics, next-month excess-return target, in-sample flag.

    python3 build_panel.py

Takes a few minutes (the trailing-12-month beta loop). The panel is the input to refit.py; backtest.py needs only results/predictions_rf.csv and the prices."""
import time, pandas as pd
from adr_longshort import load_prices, load_rf, build_panel, add_target, research_mask, ROOT
t0 = time.time(); P = load_prices(); print(f"daily rows {len(P):,}, names {P.ticker.nunique()}")
M, stats = build_panel(P, verbose=True); print("price cleaning:", stats, f"[{time.time()-t0:.0f}s]")
_, rfm = load_rf(); M = add_target(M, rfm); M["in_sample"] = research_mask(M)                       # ranks are added by the consumer after reading the file (refit.py), as in the research pipeline
M = M[M.ym >= "2000-01"]
n = M[M.in_sample].groupby("ym").ticker.nunique(); print("in-sample names per December:", {str(k): int(v) for k, v in n[n.index.month == 12].items()})
print(f"panel rows {len(M):,}, in-sample {int(M.in_sample.sum()):,}, names {M.ticker.nunique()}")
M.to_csv(f"{ROOT}/results/panel.csv", index=False); print(f"results/panel.csv written [{time.time()-t0:.0f}s]")
