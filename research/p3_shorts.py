"""Phase 3.3 borrow-aware short ranking, 3.4 inverse-vol shorts with caps, 3.5 combinations. Tails and a +200% shock test.
Rerun after research/out/p4_borrow_panel.csv exists to add the observed/imputed-fee and availability variants."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, "research"); from p3_lib import *
from trials import log
def adj_sched(sched): return lambda gs: gs.adv.map(lambda a: tier(a, sched)) / 12
V = {"baseline (EW shorts, forecast rank)": dict(H=dict(), book=dict(long_kind="stocks")),
     "borrow-aware, base schedule": dict(H=dict(score_adj=adj_sched("base")), book=dict(long_kind="stocks")),
     "borrow-aware, harsh schedule": dict(H=dict(score_adj=adj_sched("harsh")), book=dict(long_kind="stocks")),
     "inverse-vol shorts (cap 2x EW, 15%)": dict(H=dict(), book=dict(long_kind="stocks", short_w="invvol"))}
P4 = "research/out/p4_borrow_panel.csv"
if os.path.exists(P4):
    bp = pd.read_csv(P4); bp["ym"] = pd.PeriodIndex(bp.ym, freq="M"); fee = bp.set_index(["ticker", "ym"]).fee_best
    unav = set(map(tuple, bp[bp.available_flag == 0][["ticker", "ym"]].itertuples(index=False))) if "available_flag" in bp else set()
    def adj_fee(gs):
        ym = gs.ym.iloc[0]; return pd.Series([fee.get((t, ym), np.nan) for t in gs.ticker], index=gs.index).fillna(gs.adv.map(lambda a: tier(a, "harsh"))) / 12
    V["borrow-aware, P4 fee"] = dict(H=dict(score_adj=adj_fee), book=dict(long_kind="stocks"))
    V["borrow-aware, P4 fee + skip unavailable"] = dict(H=dict(score_adj=adj_fee, avail=unav), book=dict(long_kind="stocks"))
for hd in ["VW", "FXI"]:
    V[f"short + {hd} x1, inverse-vol shorts"] = dict(H=dict(), book=dict(long_kind="hedge", hedge=hd, short_w="invvol"))
    V[f"short + {hd} x1, borrow-aware base + inverse-vol"] = dict(H=dict(score_adj=adj_sched("base")), book=dict(long_kind="hedge", hedge=hd, short_w="invvol"))
V["borrow-aware base + inverse-vol (long leg)"] = dict(H=dict(score_adj=adj_sched("base")), book=dict(long_kind="stocks", short_w="invvol"))
def shock(m, k=None):
    """Largest short +200% on one day mid-month: loss = 2 x position; Reg T maintenance after the shock (long 25%, short 30%)."""
    rows = []
    for dt, r in m[m.exposure > 0].iterrows():
        pos = r.max_short if k is None else r.dollars / k; eq = r.eq0; loss = 2 * pos; L = r.long_val; S = r.dollars + 2 * pos
        rows.append(dict(loss_pct=loss / eq * 100, maint_after=(0.25 * L + 0.30 * S) / (eq - loss)))
    x = pd.DataFrame(rows); return x.loss_pct.mean(), x.loss_pct.max(), (x.maint_after > 1).sum()
rows = []; base_q = None
for name, v in V.items():
    H = make_holdings(**v["H"]); bk = build_book(H, **v["book"]); u, q = unit_series(bk); s = unit_stats(q)
    if base_q is None: base_q = q
    dl = (q.net_base - base_q.net_base.reindex(q.index)).dropna(); s["paired_t_vs_base"] = dl.mean() / dl.std() * np.sqrt(len(dl)) if dl.std() > 0 else 0
    d, m, c = simulate(bk); a = acct_stats(d, m); dh, mh, _ = simulate(bk, borrow="harsh"); ah = acct_stats(dh, mh)
    sm, sx, scall = shock(m); s5m, s5x, s5call = shock(m, 5)
    rows.append(dict(variant=name, **s, **{f"acct_{k}": x for k, x in a.items()}, acct_harsh_sharpe=ah["sharpe"], worst_name_loss_pct=c.loss_pct.min() if len(c) else np.nan,
                     avg_max_short_w=q.max_w.mean(), shock_loss_avg_pct=sm, shock_loss_max_pct=sx, shock_call_months=scall, shock5_loss_max_pct=s5x, shock5_call_months=s5call))
    log("P3", f"3.3-3.5 variant: {name}", f"unit net {s['net_base_sharpe']:.2f} (t vs base {s['paired_t_vs_base']:.2f}); acct {a['sharpe']:.2f} harsh {ah['sharpe']:.2f}; worst day {a['worst_day_pct']:.1f}%", "research/out/p3_shorts.csv")
    print(name, round(s["net_base_sharpe"], 2), round(a["sharpe"], 2), flush=True)
T = pd.DataFrame(rows).set_index("variant"); T.to_csv("research/out/p3_shorts.csv"); pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
print(T[["net_base_sharpe", "net_harsh_sharpe", "paired_t_vs_base", "gross_ret", "borrow_pa", "cost_pa", "net_21_26", "acct_final_equity", "acct_sharpe", "acct_harsh_sharpe", "acct_maxdd_pct", "acct_worst_month_pct", "acct_worst_day_pct", "worst_name_loss_pct", "avg_max_short_w", "shock_loss_avg_pct", "shock_loss_max_pct", "shock_call_months", "shock5_loss_max_pct", "shock5_call_months"]].round(2).to_string())
