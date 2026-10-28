"""Phase 3.1: short only, then each hedge. Unit book + $500k account. Attribution against the VW universe."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "research"); from p3_lib import *
from trials import log
H = make_holdings()
BOOKS = {"short only (naked)": dict(long_kind="none"), "short + current long leg (baseline)": dict(long_kind="stocks")}
for hd in ["PGJ", "KWEB", "FXI", "MCHI", "VW", "EW"]:
    BOOKS[f"short + {hd} x1"] = dict(long_kind="hedge", hedge=hd); BOOKS[f"short + {hd} beta-matched"] = dict(long_kind="hedge", hedge=hd, beta_match=True)
U = {}; Q = {}; rows = []; yr = {}
naked_cost = None
for name, kw in BOOKS.items():
    bk = build_book(H, **kw); u, q = unit_series(bk); Q[name] = q; s = unit_stats(q)
    if name.startswith("short only"): naked_cost = q.cost
    s["hedge_cost_pa"] = (q.cost - naked_cost.reindex(q.index)).mean() * 1200; s["tracking_err"] = (q.L - q.S).std() * np.sqrt(12) * 100
    c = q.loc["2013-09":]; s["net_common_13_26"] = c.net_base.mean() / c.net_base.std() * np.sqrt(12)
    d, m, _ = simulate(bk); a = acct_stats(d, m); dh, mh, _ = simulate(bk, borrow="harsh"); ah = acct_stats(dh, mh)
    rows.append(dict(book=name, **s, **{f"acct_{k}": v for k, v in a.items()}, acct_harsh_sharpe=ah["sharpe"], acct_harsh_final=ah["final_equity"]))
    yr[name] = q.net_base.groupby(q.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    d.to_csv(f"research/out/p3_daily_{name.split()[2] if 'x1' in name or 'beta' in name else name.split()[0]}_{'beta' if 'beta' in name else 'x1'}.csv")
    log("P3", f"3.1 book: {name}", f"unit net {s['net_base_sharpe']:.2f} harsh {s['net_harsh_sharpe']:.2f}; acct {a['sharpe']:.2f} final {a['final_equity']:,.0f} maxdd {a['maxdd_pct']:.1f}", "research/out/p3_hedges.csv")
    print(name, round(s["net_base_sharpe"], 2), round(a["sharpe"], 2), flush=True)
T = pd.DataFrame(rows).set_index("book"); T.to_csv("research/out/p3_hedges.csv"); pd.DataFrame(yr).round(1).to_csv("research/out/p3_hedges_by_year.csv")
# attribution: VW universe as the market
vw = Q["short + VW x1"].L; att = []
for name, q in Q.items():
    S, L = q.S, q.L; ok = vw.reindex(S.index).notna(); bS = np.cov(S[ok], vw[S.index][ok])[0, 1] / vw[S.index][ok].var()
    short_alpha = -(S - bS * vw.reindex(S.index)); mkt = L - bS * vw.reindex(S.index)
    row = dict(book=name, beta_short_vw=bS, short_alpha_pa=short_alpha.mean() * 1200, hedge_or_long_minus_shortbeta_pa=mkt.mean() * 1200, total_gross_pa=q.gross.mean() * 1200, costs_borrow_pa=(q.cost + q.b_base).mean() * 1200, net_pa=q.net_base.mean() * 1200)
    if "long leg" in name:
        bL = np.cov(L[ok], vw[L.index][ok])[0, 1] / vw[L.index][ok].var(); row.update(beta_long_vw=bL, long_alpha_pa=(L - bL * vw.reindex(L.index)).mean() * 1200, market_pa=((bL - bS) * vw.reindex(L.index)).mean() * 1200)
    att.append(row)
pd.DataFrame(att).set_index("book").to_csv("research/out/p3_attribution.csv")
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
cols = ["net_base_sharpe", "net_harsh_sharpe", "net_common_13_26", "gross_ret", "vol", "maxdd", "worst_month", "hedge_cost_pa", "tracking_err", "hedge_ratio", "net_10_15", "net_16_20", "net_21_26", "acct_final_equity", "acct_cagr_pct", "acct_sharpe", "acct_harsh_sharpe", "acct_maxdd_pct", "acct_worst_month_pct", "acct_beta_pgj_m", "acct_sharpe_21_26"]
print(T[cols].round(2).to_string()); print(pd.DataFrame(att).set_index("book").round(2).to_string()); print(pd.DataFrame(yr).round(0).to_string())
