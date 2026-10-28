"""Phase 4.3: the realistic account. Grid: trading costs (flat 20 bp / per-name at 0.5x and 1x the high-low spread, + tick floor, commission, impact)
x borrow (assumed base schedule / observed-or-imputed IBKR fee) x availability (all / skip unavailable). Usage: python research/p4_realistic.py <pred_path> <tag>"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "research"); from realistic import load_real
from trials import log
pred, tag = sys.argv[1], sys.argv[2]; rows = []
grid = [("flat 20bp, base borrow (backtest as built)", dict(costs=False)),
        ("per-name costs 0.5x spread, base borrow", dict(costs=True, spread_scale=0.5)), ("per-name costs 1x spread, base borrow", dict(costs=True, spread_scale=1.0)),
        ("flat 20bp, IBKR fee (obs/imputed)", dict(costs=False, borrow="p4")), ("flat 20bp, IBKR fee, skip unavailable", dict(costs=False, borrow="p4", skip_unavailable=True)),
        ("per-name 0.5x, IBKR fee", dict(costs=True, spread_scale=0.5, borrow="p4")), ("per-name 0.5x, IBKR fee, skip unavailable", dict(costs=True, spread_scale=0.5, borrow="p4", skip_unavailable=True)),
        ("per-name 1x, IBKR fee, skip unavailable", dict(costs=True, spread_scale=1.0, borrow="p4", skip_unavailable=True))]
for lab, kw in grid:
    ns = load_real(pred, **kw); d, m = ns["simulate"](); s = ns["stats"](d, m)
    r = dict(variant=lab, final=s["final_equity"], cagr=s["cagr_pct"], vol=s["vol_pct"], sharpe=s["sharpe"], sharpe_m=s["sharpe_monthly"], maxdd=s["maxdd_pct"], worst_m=s["worst_month_pct"],
             fees=s["fees_usd"], borrow=s["borrow_usd"], interest=s["interest_usd"], cost_bp_traded=s["fees_usd"] / m.traded.sum() * 1e4, borrow_rate=s["borrow_rate_pct"],
             sharpe_2021_26=d.ret.loc["2021":].pipe(lambda x: x.mean() / x.std() * np.sqrt(252)), sharpe_2010_15=d.ret.loc["2010":"2015"].pipe(lambda x: x.mean() / x.std() * np.sqrt(252)))
    rows.append(r); log("P4", f"realistic account [{tag}]: {lab}", f"sharpe {r['sharpe']:.2f} final {r['final']:,.0f}", f"research/out/p4_realistic_{tag}.csv"); print(lab, round(r["sharpe"], 2), flush=True)
T = pd.DataFrame(rows).set_index("variant"); T.to_csv(f"research/out/p4_realistic_{tag}.csv"); pd.set_option("display.width", 250); print(T.round(2).to_string())
