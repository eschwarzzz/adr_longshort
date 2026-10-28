"""Phase 0.6 (part 2): unit book and $500k account for each corrected-baseline variant. Writes research/out/p0_accounts.csv."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "research"); from account import load
from trials import log
S = pd.read_csv("results/predictions_rf.csv"); Q = pd.read_csv("research/out/p0_pred_shipped.csv")
c = S.merge(Q, on=["ticker", "ym"], suffixes=("", "_q")); print(f"shipped variant vs shipped forecasts: {len(c)}/{len(S)} rows matched, max |diff| {(c.p_RF - c.p_RF_q).abs().max():.1e}")
rows = []
for tag in ["shipped", "fix_price", "fix_next", "fix_adv", "corrected"]:
    B = load(f"research/out/p0_pred_{tag}.csv"); d, m = B.simulate(); s = B.stats(d, m)
    ls = (m.long_ret - m.short_ret).dropna(); u = dict(unit_sharpe=ls.mean() / ls.std() * np.sqrt(12), unit_ret=ls.mean() * 1200, long_ret=m.long_ret.mean() * 1200, short_ret=m.short_ret.mean() * 1200)
    pg = B.B["PGJ"].reindex(m.index.to_period("M")).values; mr = m.ret.values; ok = ~np.isnan(pg)
    s["beta_pgj"] = np.cov(mr[ok], pg[ok])[0, 1] / np.var(pg[ok]); s["sharpe_2021_26"] = d.ret.loc["2021":].pipe(lambda r: r.mean() / r.std() * np.sqrt(252))
    s["sharpe_2010_15"] = d.ret.loc["2010":"2015"].pipe(lambda r: r.mean() / r.std() * np.sqrt(252)); s["sharpe_2016_20"] = d.ret.loc["2016":"2020"].pipe(lambda r: r.mean() / r.std() * np.sqrt(252))
    rows.append(dict(variant=tag, **u, **{k: s[k] for k in ["final_equity", "cagr_pct", "vol_pct", "sharpe", "sharpe_monthly", "maxdd_pct", "worst_month_pct", "beta_pgj", "sharpe_2010_15", "sharpe_2016_20", "sharpe_2021_26", "avg_side_pct", "n_short", "fees_usd", "borrow_usd", "interest_usd", "call_days"]}))
    m.to_csv(f"research/out/p0_monthly_{tag}.csv"); d.to_csv(f"research/out/p0_daily_{tag}.csv")
    log("P0", f"account on baseline variant {tag}", f"sharpe {s['sharpe']:.2f}, unit {u['unit_sharpe']:.2f}, final {s['final_equity']:,.0f}", f"research/out/p0_monthly_{tag}.csv")
T = pd.DataFrame(rows).set_index("variant"); T.to_csv("research/out/p0_accounts.csv"); pd.set_option("display.width", 250); print(T.round(2).T.to_string())
