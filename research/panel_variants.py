"""Rebuild the research panel under a variant: nocap (keep serial reverse-splitters), noclean (raw returns in features and targets).
    python research/panel_variants.py nocap|noclean   -> research/out/panel_<variant>.csv"""
import sys, pandas as pd
sys.path.insert(0, "."); import adr_longshort.features as fe
from adr_longshort import load_prices, load_rf
v = sys.argv[1]
if v == "nocap": fe.MAX_ADJ_PRICE = float("inf")
if v == "noclean":
    def raw(P, **k):
        P = P.sort_values(["ticker", "date"]).reset_index(drop=True); P["ret"] = P.groupby("ticker").close.pct_change(); return P, {}
    fe.clean = raw
M, st = fe.build_panel(load_prices()); _, rfm = load_rf(); M = fe.add_target(M, rfm); M["in_sample"] = fe.research_mask(M); M = M[M.ym >= "2000-01"]
print(v, "panel rows", len(M), "in-sample", int(M.in_sample.sum()), "names", M.ticker.nunique()); M.to_csv(f"research/out/panel_{v}.csv", index=False)
