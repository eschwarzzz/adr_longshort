"""Standard report for any forecast column, so every phase reports the same things against the corrected baseline.
    full_report(pred_path, col="p_RF", baseline=("research/out/p0_pred_corrected.csv", "p_RF"), sim=dict(...), holdings_fn=None) -> dict
Unit book (monthly, $1 a side, from exret_next) and the $500k account (daily, backtest.py's simulator via research/account.py).
holdings_fn(month_df, col) -> dict(long, short, adv_short) lets a construction variant replace adr_longshort.holdings in the unit book."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "research")
from adr_longshort import load_predictions, load_benchmarks, load_rf, holdings
from account import load as load_account
BASE = [(25e6, .01), (10e6, .04), (3e6, .12), (0, .35)]; HARSH = [(25e6, .02), (10e6, .08), (3e6, .25), (0, .60)]; C = 0.002
tier = lambda a, s: next(r for lo, r in s if a >= lo)
sr = lambda x: np.nan if len(x.dropna()) < 3 or x.std() == 0 else x.mean() / x.std() * np.sqrt(12)
B = load_benchmarks(); _, RFM = load_rf()
def unit_book(F, col, holdings_fn=None):
    L = F[F.liquid]; rows = []; p5 = p1 = set()
    for ym, g in L.groupby("ym"):
        h = (holdings_fn or (lambda g, c: holdings(g, c)))(g, col)
        if h is None: continue
        lo = g[g.ticker.isin(h["long"])]; sh = g[g.ticker.isin(h["short"])]; s5, s1 = set(h["long"]), set(h["short"])
        wl = h.get("w_long"); ws = h.get("w_short")
        lr = np.average(lo.set_index("ticker").exret_next.reindex(h["long"]).fillna(0), weights=wl) if len(lo) else np.nan
        srt = np.average(sh.set_index("ticker").exret_next.reindex(h["short"]).fillna(0), weights=ws) if len(sh) else np.nan
        vw = np.average(g.exret_next.fillna(0), weights=g.adv)
        rows.append(dict(ym=ym, L=lr, S=srt, EW=g.exret_next.mean(), VW=vw, PGJ=B["PGJ"].get(ym + 1, np.nan) - RFM.get(ym + 1, np.nan), KWEB=B["KWEB"].get(ym + 1, np.nan) - RFM.get(ym + 1, np.nan),
                         to5=1 - len(s5 & p5) / len(s5) if p5 else 1, to1=1 - len(s1 & p1) / len(s1) if p1 else 1, bb=np.mean([tier(a, BASE) for a in sh.adv]) / 12, bh=np.mean([tier(a, HARSH) for a in sh.adv]) / 12, nS=len(sh), nL=len(lo)))
        p5, p1 = s5, s1
    q = pd.DataFrame(rows).set_index("ym"); q["LS"] = q.L - q.S; q["LS_net"] = q.LS - C * 2 * (q.to5 + q.to1) - q.bb; q["LS_net_h"] = q.LS - C * 2 * (q.to5 + q.to1) - q.bh; return q
def beta(x, b): ok = x.notna() & b.notna(); return np.cov(x[ok], b[ok])[0, 1] / b[ok].var()
def full_report(pred_path, col="p_RF", baseline=("research/out/p0_pred_corrected.csv", "p_RF"), sim=None, holdings_fn=None, account=True):
    F = load_predictions(pred_path); q = unit_book(F, col, holdings_fn)
    out = dict(months=len(q), n_long=q.nL.mean(), n_short=q.nS.mean(), unit_ret=q.LS.mean() * 1200, unit_vol=q.LS.std() * np.sqrt(12) * 100, unit_sharpe=sr(q.LS), unit_net_sharpe=sr(q.LS_net), unit_net_harsh=sr(q.LS_net_h),
               long_ret=q.L.mean() * 1200, short_ret=q.S.mean() * 1200, long_minus_pgj_net=sr(q.L - q.PGJ - C * 2 * q.to5), beta_pgj=beta(q.LS, q.PGJ), beta_vw=beta(q.LS, q.VW), beta_short_pgj=beta(q.S, q.PGJ),
               unit_maxdd=((1 + q.LS_net).cumprod() / (1 + q.LS_net).cumprod().cummax() - 1).min() * 100, turnover_long=q.to5.mean(), turnover_short=q.to1.mean(),
               **{f"unit_sharpe_{a}_{b[-2:]}": sr(q.LS.loc[a:b]) for a, b in [("2010", "2015"), ("2016", "2020"), ("2021", "2026")]})
    if baseline:
        qb = unit_book(load_predictions(baseline[0]), baseline[1]); dlt = (q.LS_net - qb.LS_net.reindex(q.index)).dropna(); out["paired_t_vs_base"] = dlt.mean() / dlt.std() * np.sqrt(len(dlt)) if dlt.std() > 0 else 0.0
    if account and holdings_fn is None:
        A = load_account(pred_path, col); d, m = A.simulate(**(sim or {})); s = A.stats(d, m); pg = B["PGJ"].reindex(m.index.to_period("M")).values; ok = ~np.isnan(pg)
        out.update({f"acct_{k}": s[k] for k in ["final_equity", "cagr_pct", "vol_pct", "sharpe", "sharpe_monthly", "maxdd_pct", "worst_month_pct", "avg_side_pct", "fees_usd", "borrow_usd", "interest_usd", "call_days"]})
        out["acct_beta_pgj"] = np.cov(m.ret.values[ok], pg[ok])[0, 1] / np.var(pg[ok]); out["acct_sharpe_2021_26"] = d.ret.loc["2021":].pipe(lambda r: r.mean() / r.std() * np.sqrt(252))
        out["acct_worst_day_pct"] = d.ret.min() * 100
    return out, q
