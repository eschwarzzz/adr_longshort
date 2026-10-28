"""Reuses backtest.py's account simulator (Part 3) with switches, without the Monte Carlo, writing nothing to results/.
    load(pred_path=None, col="p_RF", clean_mode="repo") -> namespace with simulate, stats, hold_all, U, VOL, RET, ...
clean_mode: "repo" (shipped cleaning) | "none" (raw close-to-close returns for the P&L)."""
import sys, types, numpy as np, pandas as pd  # noqa
sys.path.insert(0, ".")
SRC = open("backtest.py").read()
def load(pred_path=None, col="p_RF", clean_mode="repo", cash_haircut=0.005, proceeds_haircut=0.0125):
    s = SRC.split("t0 = time.time(); H_full")[0]
    s = s.replace('R = f"{ROOT}/results"', 'R = "research/out"').replace("QUICK = \"--quick\" in sys.argv", "QUICK = False")
    s = s.replace("F = load_predictions()", f"F = load_predictions({pred_path!r}); F['p_RF'] = F[{col!r}]")
    s = s.replace("S1.to_csv(", "(lambda *a, **k: None)(").replace("QR[MODEL].reset_index().assign(strategy=MODEL).to_csv(", "(lambda *a, **k: None)(").replace("S2.to_csv(", "(lambda *a, **k: None)(")
    s = s.replace("print(", "_p(")
    if clean_mode == "none": s = s.replace("P, cstats = clean(P)", "P = P.sort_values(['ticker','date']).reset_index(drop=True); P['ret'] = P.groupby('ticker').close.pct_change(); cstats = {}")
    s = s.replace("rf.loc[d] - 0.005", f"rf.loc[d] - {cash_haircut}").replace("rf.loc[d] - 0.0125", f"rf.loc[d] - {proceeds_haircut}")
    ns = {"__name__": "bt", "_p": lambda *a, **k: None}; exec(compile(s, "backtest_harness", "exec"), ns); return types.SimpleNamespace(**ns)

def load_ns(pred_path=None, col="p_RF", ret_override=None, **kw):
    """Like load(), but returns the live namespace dict; with ret_override (date x ticker daily returns) the P&L, unit series and sizing
    volatility are rebuilt from those returns."""
    s = SRC.split("t0 = time.time(); H_full")[0]
    s = s.replace('R = f"{ROOT}/results"', 'R = "research/out"').replace("QUICK = \"--quick\" in sys.argv", "QUICK = False")
    s = s.replace("F = load_predictions()", f"F = load_predictions({pred_path!r}); F['p_RF'] = F[{col!r}]")
    s = s.replace("S1.to_csv(", "(lambda *a, **k: None)(").replace("QR[MODEL].reset_index().assign(strategy=MODEL).to_csv(", "(lambda *a, **k: None)(").replace("S2.to_csv(", "(lambda *a, **k: None)(").replace("print(", "_p(")
    ns = {"__name__": "bt", "_p": lambda *a, **k: None}; exec(compile(s, "backtest_harness", "exec"), ns)
    if ret_override is not None:
        ns["RET"] = ret_override.reindex(index=ns["DAYS"]); ns["U"] = ns["unit_series"](ns["H"]); ns["VOL"] = ns["sizing_vol"](ns["U"])
    return ns
