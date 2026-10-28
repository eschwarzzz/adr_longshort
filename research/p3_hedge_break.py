"""Phase 3.2: why the long-leg-vs-ETF relationship changed after 2021. Year-by-year composition and beta of each leg."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "research"); from p3_lib import *
from trials import log
Pn = pd.read_csv("research/out/p0_panel_corrected.csv", usecols=["ticker", "ym", "price", "price_real", "adv"]); Pn["ym"] = pd.PeriodIndex(Pn.ym, freq="M"); Pn = Pn.set_index(["ticker", "ym"])
first = RET.apply(lambda s: s.first_valid_index())
H = make_holdings(); rows = []; Ld = []; Sd = []; VWd = []
for ym, h in H.items():
    days = A.month_days(ym)
    if len(days) == 0: continue
    sr_, _ = basket_daily(h["short"], np.full(len(h["short"]), 1 / len(h["short"])), days); lr_, _ = basket_daily(h["long"], np.full(len(h["long"]), 1 / len(h["long"])), days)
    names, w = universe_basket(G[ym], "VW"); vr_, _ = basket_daily(names, w, days); Sd.append(sr_); Ld.append(lr_); VWd.append(vr_)
    end = ym.to_timestamp(how="end")
    for side in ("short", "long"):
        for t in h[side]:
            age = (end - first.get(t, end)).days / 30.4 if pd.notna(first.get(t)) else np.nan
            pr = Pn.price_real.get((t, ym), np.nan); rows.append(dict(ym=ym, year=ym.year, side=side, ticker=t, age_m=age, price=pr, adv=G[ym].set_index("ticker").adv.get(t, np.nan)))
C = pd.DataFrame(rows); Sd, Ld, VWd = pd.concat(Sd), pd.concat(Ld), pd.concat(VWd); pg = ETF["PGJ"].reindex(Sd.index)
def yb(x, b):
    ok = x.notna() & b.notna(); x, b = x[ok], b[ok]; be = np.cov(x, b)[0, 1] / b.var(); r2 = np.corrcoef(x, b)[0, 1] ** 2; return be, 1 - r2, np.corrcoef(x, b)[0, 1]
out = []
for y in range(2010, 2027):
    s = C[(C.year == y) & (C.side == "short")]; l = C[(C.year == y) & (C.side == "long")]; idx = Sd.index.year == y
    if not idx.any(): continue
    bS, idS, _ = yb(Sd[idx], pg[idx]); bL, idL, cL = yb(Ld[idx], pg[idx]); bSv, idSv, _ = yb(Sd[idx], VWd[idx])
    ann = lambda r: ((1 + r[idx]).prod() - 1) * 100
    out.append(dict(year=y, shorts_lt24m_pct=(s.age_m < 24).mean() * 100, shorts_lt12m_pct=(s.age_m < 12).mean() * 100, short_med_adv_m=s.adv.median() / 1e6, short_med_price=s.price.median(), long_med_adv_m=l.adv.median() / 1e6,
                    short_beta_pgj=bS, short_idio_share=idS, short_beta_vw=bSv, long_beta_pgj=bL, long_corr_pgj=cL, long_idio_share=idL,
                    short_ret=ann(Sd), long_ret=ann(Ld), pgj_ret=ann(pg.fillna(0)), vw_ret=ann(VWd), long_minus_vw=ann(Ld) - ann(VWd), long_minus_pgj=ann(Ld) - ann(pg.fillna(0)), pgj_minus_short=ann(pg.fillna(0)) - ann(Sd), long_minus_short=ann(Ld) - ann(Sd)))
T = pd.DataFrame(out).set_index("year"); T.to_csv("research/out/p3_hedge_break.csv")
P1, P2 = T.loc[2010:2020], T.loc[2021:2026]
summ = pd.DataFrame({"2010-20": P1.mean(), "2021-26": P2.mean()}); summ.to_csv("research/out/p3_hedge_break_summary.csv")
log("P3", "3.2 hedge-break diagnostics by year", f"shorts<24m {P1.shorts_lt24m_pct.mean():.0f}% -> {P2.shorts_lt24m_pct.mean():.0f}%; short beta PGJ {P1.short_beta_pgj.mean():.2f} -> {P2.short_beta_pgj.mean():.2f}", "research/out/p3_hedge_break.csv")
pd.set_option("display.width", 300); print(T.round(2).to_string()); print(summ.round(2).to_string())
