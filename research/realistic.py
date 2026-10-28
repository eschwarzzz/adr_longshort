"""Phase 4: the account with per-name trading costs and (optionally) per-name borrow and availability, via hooks patched into backtest.py's simulator.
    ns = load_real(pred_path, col="p_RF", costs=True, impact_k=0.5, borrow=None|"p4", skip_unavailable=False, borrow_floor_sched=None)
    d, m = ns["simulate"](); s = ns["stats"](d, m)
Costs per side (traded dollars): half-spread (Abdi-Ranaldo, floored at half a tick on the traded price) + $0.0035/share + impact_k * sigma_daily * sqrt(order / ADV), capped at 10%.
Borrow: research/out/p4_borrow_panel.csv fee_best (annual, decimal or percent auto-detected); missing -> the tiered base schedule.
skip_unavailable: short candidates flagged unavailable (available_flag == 0/False) are skipped and the next-ranked name in the $3m pool is taken."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "research")
from adr_longshort import holdings as _holdings
SRC = open("backtest.py").read()
def load_real(pred_path, col="p_RF", costs=True, impact_k=0.5, borrow=None, skip_unavailable=False, fallback_sched="base", spread_scale=1.0, borrow_aware=False):
    s = SRC.split("t0 = time.time(); H_full")[0]
    s = s.replace('R = f"{ROOT}/results"', 'R = "research/out"').replace("QUICK = \"--quick\" in sys.argv", "QUICK = False")
    s = s.replace("F = load_predictions()", f"F = load_predictions({pred_path!r}); F['p_RF'] = F[{col!r}]")
    s = s.replace("S1.to_csv(", "(lambda *a, **k: None)(").replace("QR[MODEL].reset_index().assign(strategy=MODEL).to_csv(", "(lambda *a, **k: None)(").replace("S2.to_csv(", "(lambda *a, **k: None)(").replace("print(", "_p(")
    s = s.replace("h = holdings(g, col or f\"p_{model}\", short_floor=short_floor)", "h = HOLD(g, col or f\"p_{model}\", short_floor=short_floor, ym=ym)")
    s = s.replace("traded = sum(abs(tgt.get(t, 0.0) - pos.get(t, 0.0)) for t in set(tgt) | set(pos)); fee0 = traded * cost",
                  "traded = sum(abs(tgt.get(t, 0.0) - pos.get(t, 0.0)) for t in set(tgt) | set(pos)); fee0 = sum(abs(tgt.get(t, 0.0) - pos.get(t, 0.0)) * COSTFN(t, ym, abs(tgt.get(t, 0.0) - pos.get(t, 0.0)), cost) for t in set(tgt) | set(pos))")
    s = s.replace("rates = {t: tier(a, sched) for t, a in zip(h[\"short\"], h[\"adv_short\"])}", "rates = {t: BORROWFN(t, ym, a, sched) for t, a in zip(h[\"short\"], h[\"adv_short\"])}")
    assert s.count("HOLD(") >= 1 and "COSTFN(" in s and "BORROWFN(" in s, "patch points not found in backtest.py"
    SP = pd.read_csv("research/out/p4_spreads.csv"); SP["ym"] = pd.PeriodIndex(SP.ym, freq="M"); SP = SP.set_index(["ticker", "ym"])
    PR = pd.read_csv(pred_path); PR["ym"] = pd.PeriodIndex(PR.ym, freq="M"); ADV = PR.set_index(["ticker", "ym"]).adv
    def COSTFN(t, ym, order, flat):
        if not costs: return flat
        try: r = SP.loc[(t, ym)]; base = (max(r.half_spread_bp * spread_scale, r.tick_half_bp) + r.comm_bp) / 1e4 if pd.notna(r.half_spread_bp) else flat; sd = r.sigma_d if pd.notna(r.sigma_d) else 0.04
        except KeyError: base, sd = flat, 0.04
        adv = ADV.get((t, ym), np.nan); imp = impact_k * sd * np.sqrt(order / adv) if adv and adv > 0 else 0.0
        return float(min(base + imp, 0.10))
    BP = None
    if borrow == "p4":
        import os; BP = pd.read_csv("research/out/p4_borrow_panel_all.csv" if os.path.exists("research/out/p4_borrow_panel_all.csv") else "research/out/p4_borrow_panel.csv"); BP["ym"] = pd.PeriodIndex(BP.ym, freq="M"); BP = BP.drop_duplicates(["ticker", "ym"]).set_index(["ticker", "ym"])
        if BP.fee_best.dropna().median() > 1.5: BP["fee_best"] = BP.fee_best / 100          # percent -> decimal
    def BORROWFN(t, ym, adv, sched):
        if BP is not None:
            try:
                v = BP.loc[(t, ym), "fee_best"]
                if pd.notna(v): return float(v)
            except KeyError: pass
        for lo, r in sched:
            if adv >= lo: return r
        return sched[-1][1]
    def HOLD(g, c, short_floor=3e6, ym=None):
        if not ((skip_unavailable or borrow_aware) and BP is not None): return _holdings(g, c, short_floor=short_floor)
        av = [BP.available_flag.get((t, ym), 1) for t in g.ticker]; ok = pd.Series([bool(a) if pd.notna(a) else True for a in av], index=g.index)
        if not skip_unavailable: ok[:] = True
        h = _holdings(g, c, short_floor=short_floor)
        if h is None: return None
        fee = pd.Series([BORROWFN(t, ym, a, BORROW_BASE) for t, a in zip(g.ticker, g.adv)], index=g.index)
        score = g[c] + (fee / 12 if borrow_aware else 0.0)
        pool = g.assign(_s=score)[(g.adv >= max(short_floor, 0)) & ok].sort_values("_s"); k = len(h["short"]); short = pool.head(k)
        if len(short) < 5: return None
        h["short"] = list(short.ticker); h["adv_short"] = short.adv.values; return h
    BORROW_BASE = [(25e6, 0.01), (10e6, 0.04), (3e6, 0.12), (0, 0.35)]
    ns = {"__name__": "bt", "_p": lambda *a, **k: None, "COSTFN": COSTFN, "BORROWFN": BORROWFN, "HOLD": HOLD}
    exec(compile(s, "backtest_real", "exec"), ns); return ns
