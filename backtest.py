"""Backtest of the random-forest Chinese ADR long-short, from results/predictions_rf.csv and the daily prices.

    python3 backtest.py              # everything below, about 10 minutes
    python3 backtest.py --quick      # quintile statistics and borrow variants only, a few seconds

Part 1, quintile statistics (results/quintile_stats_liquid_ew.csv, quintile_returns_rf.csv): equal-weight quintiles on the liquid universe
(>= $1m a day, 25 or more names), long-short gross and net of 20 bp a side plus 3% borrow, sub-periods.
Part 2, borrow variants (results/borrow_variants_rf.csv): flat and volume-tiered borrow schedules, short-leg floors, ETF-hedged long leg.
Part 3, the $500,000 account (results/bt500k_rf_*): daily P&L on the monthly holdings with the short leg above $10m a day, exposure per side =
min(80%, 10% / sigma) x equity, 20 bp a side, borrow per name by volume, cash interest, Reg T margin; variants; block bootstrap of Sharpe,
P&L and drawdown; random-sort null; stale forecasts; deflated Sharpe. Then the same for a fresh account over the last five years.
Random seeds are fixed, so the output reproduces the tables in docs/STATS.md."""
import os, sys, json, time, warnings, numpy as np, pandas as pd
from scipy.stats import norm, skew, kurtosis
from adr_longshort import load_prices, load_rf, load_benchmarks, load_predictions, clean, holdings, sizing_vol, ROOT
warnings.filterwarnings("ignore"); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
R = f"{ROOT}/results"; QUICK = "--quick" in sys.argv; rng = np.random.default_rng(42)
CAPITAL = 500_000.0; TARGET = 0.10; CAP = 0.80; COST = 0.002; MIN_N = 25; SHORT_FLOOR = 1e7; MODEL = "RF"; ALT = "ENS"; ALT_NAME = "Ensemble"; TAG = "bt500k_rf"
BORROW = {"base": [(25e6, 0.01), (10e6, 0.04), (3e6, 0.12), (0, 0.35)], "harsh": [(25e6, 0.02), (10e6, 0.08), (3e6, 0.25), (0, 0.60)], "flat3": [(0, 0.03)], "flat8": [(0, 0.08)], "zero": [(0, 0.0)]}
START = "2010-01"; END = "2026-08"; SUBS = {"2010-15": ("2010", "2015"), "2016-20": ("2016", "2020"), "2021-26": ("2021", "2026")}
def sharpe_m(x): return np.nan if x.std() == 0 else x.mean() / x.std() * np.sqrt(12)
def maxdd(x): c = (1 + x).cumprod(); return (c / c.cummax() - 1).min()
def tier(adv, sched):
    for lo, r in sched:
        if adv >= lo: return r
    return sched[-1][1]

F = load_predictions(); LIQ = F[F.liquid].copy(); has_alt = f"p_{ALT}" in F
B = load_benchmarks(); rf_daily, rfm = load_rf()

# ---------- Part 1: quintile statistics ----------
def quintiles(df, col):
    out = []; prev5 = prev1 = None
    for ym, g in df.groupby("ym"):
        if len(g) < MIN_N: continue
        g = g.copy(); g["q"] = pd.qcut(g[col].rank(method="first"), 5, labels=False) + 1; r = g.groupby("q").exret_next.mean(); s5 = set(g[g.q == 5].ticker); s1 = set(g[g.q == 1].ticker)
        to5 = 1 - len(s5 & prev5) / len(s5) if prev5 else 1.0; to1 = 1 - len(s1 & prev1) / len(s1) if prev1 else 1.0; prev5, prev1 = s5, s1
        out.append(dict(ym=ym, n=len(g), **{f"q{k}": r.get(k, np.nan) for k in range(1, 6)}, ew=g.exret_next.mean(), to5=to5, to1=to1))
    return pd.DataFrame(out).set_index("ym")
def qstats(q, cost=COST, borrow=0.03):
    ls = q.q5 - q.q1; lo = q.q5 - q.ew; ls_net = ls - (cost * 2 * (q.to5 + q.to1) + borrow / 12); lo_net = lo - cost * 2 * q.to5
    d = dict(months=len(q), avg_names=q.n.mean(), ls_ret_pct=ls.mean() * 1200, ls_sharpe=sharpe_m(ls), ls_t=ls.mean() / ls.std() * np.sqrt(len(ls)), ls_net_sharpe=sharpe_m(ls_net), ls_maxdd_pct=maxdd(ls) * 100,
             q5_ret_pct=q.q5.mean() * 1200, q1_ret_pct=q.q1.mean() * 1200, ew_ret_pct=q.ew.mean() * 1200, long_excess_pct=lo.mean() * 1200, long_excess_sharpe=sharpe_m(lo), long_excess_net_sharpe=sharpe_m(lo_net),
             q5_sharpe=sharpe_m(q.q5), ew_sharpe=sharpe_m(q.ew), turnover_q5=q.to5.mean(), turnover_q1=q.to1.mean())
    for lab, (a, b) in SUBS.items(): d[f"ls_sharpe_{lab}"] = sharpe_m(ls.loc[a:b])
    return d
QS = {}; QR = {}
for name in [MODEL] + ([ALT] if has_alt else []):
    q = quintiles(LIQ, f"p_{name}"); QR[name] = q; QS[name] = qstats(q)
S1 = pd.DataFrame(QS).T; S1.to_csv(f"{R}/quintile_stats_liquid_ew.csv"); QR[MODEL].reset_index().assign(strategy=MODEL).to_csv(f"{R}/quintile_returns_rf.csv", index=False)
print("[1] Liquid universe, equal-weight quintiles, costs 20 bp a side + 3% borrow:"); print(S1.round(2).to_string())

# ---------- Part 2: borrow variants ----------
def legs_monthly(model, short_floor=0.0):
    rows = []; prev5 = prev1 = set()
    for ym, g in LIQ.groupby("ym"):
        h = holdings(g, f"p_{model}", short_floor=short_floor)
        if h is None: continue
        q5 = g[g.ticker.isin(h["long"])]; q1 = g[g.ticker.isin(h["short"])]; s5, s1 = set(h["long"]), set(h["short"])
        rows.append(dict(ym=ym, n=len(g), n1=len(q1), q5=q5.exret_next.mean(), q1=q1.exret_next.mean(), to5=1 - len(s5 & prev5) / len(s5) if prev5 else 0, to1=1 - len(s1 & prev1) / len(s1) if prev1 else 0,
                         adv1=q1.adv.median(), **{f"b_{k}": np.mean([tier(a, s) for a in q1.adv]) / 12 for k, s in BORROW.items() if k in ("base", "harsh")}))
        prev5, prev1 = s5, s1
    return pd.DataFrame(rows).set_index("ym")
def bstats(q, borrow=None, flat=None):
    ls = q.q5 - q.q1; b = q[f"b_{borrow}"] if borrow else flat / 12; net = ls - COST * 2 * (q.to5 + q.to1) - b
    return dict(months=len(q), ls_gross=ls.mean() * 1200, sharpe=sharpe_m(ls), net_ret=net.mean() * 1200, net_sharpe=sharpe_m(net), net_2021_26=sharpe_m(net.loc["2021":]), borrow_pct=float(np.mean(b)) * 1200, q1_ret=q.q1.mean() * 1200, n_short=q.n1.mean(), adv1_m=q.adv1.median() / 1e6)
out = []; q0 = legs_monthly(MODEL)
for f in [0, 0.03, 0.08, 0.15, 0.25, 0.40]: out.append(dict(variant=f"flat borrow {f:.0%}", **bstats(q0, flat=f)))
for k in ("base", "harsh"): out.append(dict(variant=f"tiered borrow ({k})", **bstats(q0, borrow=k)))
for floor, lab in [(3e6, "$3m"), (1e7, "$10m"), (2.5e7, "$25m")]:
    qq = legs_monthly(MODEL, floor)
    for k in ("base", "harsh"): out.append(dict(variant=f"short leg >= {lab}/day, tiered ({k})", **bstats(qq, borrow=k)))
for etf in ["PGJ", "KWEB"]:
    e = pd.Series(B[etf].reindex(q0.index + 1).values - rfm.reindex(q0.index + 1).values, index=q0.index); x = pd.DataFrame({"q5": q0.q5, "e": e, "to5": q0.to5}).dropna(); lo = x.q5 - x.e; net = lo - COST * 2 * x.to5 - 0.01 / 12
    out.append(dict(variant=f"long leg, short {etf}", months=len(x), ls_gross=lo.mean() * 1200, sharpe=sharpe_m(lo), net_ret=net.mean() * 1200, net_sharpe=sharpe_m(net), net_2021_26=sharpe_m(net.loc["2021":]), borrow_pct=1.0, q1_ret=e.mean() * 1200, n_short=1, adv1_m=np.nan))
S2 = pd.DataFrame(out).set_index("variant"); S2.to_csv(f"{R}/borrow_variants_rf.csv"); print("\n[2] Borrow variants (random forest):"); print(S2.round(2).to_string())
if QUICK: sys.exit()

# ---------- Part 3: the $500,000 account ----------
P = load_prices(); P, cstats = clean(P); P = P[P.date >= "2009-12-01"]; RET = P.pivot(index="date", columns="ticker", values="ret"); DAYS = RET.index
rf = rf_daily.reindex(DAYS).ffill().bfill()
def hold_all(model=MODEL, short_floor=SHORT_FLOOR, col=None):
    H = {}
    for ym, g in LIQ.groupby("ym"):
        h = holdings(g, col or f"p_{model}", short_floor=short_floor)
        if h is not None: H[ym] = dict(long=h["long"], short=h["short"], adv_short=h["adv_short"], gap=h["forecast_gap"], n=h["n_liquid"])
    return H
H = hold_all()
def leg_paths(names, days):
    r = RET.reindex(index=days, columns=names).fillna(0.0); return (1 + r).cumprod().mean(axis=1)
def month_days(ym): return DAYS[(DAYS > ym.to_timestamp(how="end")) & (DAYS <= (ym + 1).to_timestamp(how="end"))]
def unit_series(H):
    out = []
    for ym, h in H.items():
        days = month_days(ym)
        if len(days) == 0: continue
        v = leg_paths(h["long"], days) - leg_paths(h["short"], days) + 1; out.append(pd.Series(v, index=days).pct_change().fillna(v.iloc[0] - 1))
    return pd.concat(out)
U = unit_series(H); VOL = sizing_vol(U); V60 = U.rolling(60, min_periods=20).std() * np.sqrt(252); PRIOR_VOL = 0.40

def simulate(H=H, target=TARGET, cap=CAP, fixed=None, cost=COST, borrow="base", start=START, end=END, capital=CAPITAL, vol=None, long_only=False, hedge=None):
    vol = VOL if vol is None else vol; eq = capital; pos = {}; rows = []; drows = []; sched = BORROW[borrow]
    for ym in pd.period_range(pd.Period(start, "M") - 1, pd.Period(end, "M") - 1, freq="M"):
        days = month_days(ym)
        if len(days) == 0: continue
        if ym not in H:
            traded = sum(abs(v) for v in pos.values()); eq -= traded * cost; pos = {}; eq0 = eq
            for d in days:
                interest = eq * max(rf.loc[d] - 0.005, 0.0) / 252; eq += interest; rows.append((d, 0.0, 0.0, interest, 0.0, 0.0, interest, eq))
            drows.append(dict(date=ym.to_timestamp(how="end"), sigma=np.nan, exposure=0.0, dollars_side=0.0, n_long=0, n_short=0, gap=np.nan, equity_start=eq0, pnl=eq - eq0, ret=(eq - eq0) / eq0, traded=traded, borrow_rate=np.nan, borrow=0.0, fees=traded * cost, risk_fcst=0.0, long_ret=np.nan, short_ret=np.nan))
            continue
        h = H[ym]; sig = vol.loc[:ym.to_timestamp(how="end")].dropna(); sig = sig.iloc[-1] if len(sig) else PRIOR_VOL
        e = fixed if fixed is not None else min(cap, target / sig); E = e * eq
        nl, ns = len(h["long"]), len(h["short"]); tgt = {t: E / nl for t in h["long"]}
        if not long_only and hedge is None: tgt.update({t: -E / ns for t in h["short"]})
        traded = sum(abs(tgt.get(t, 0.0) - pos.get(t, 0.0)) for t in set(tgt) | set(pos)); fee0 = traded * cost
        rates = {t: tier(a, sched) for t, a in zip(h["short"], h["adv_short"])}
        r = RET.reindex(index=days, columns=list(tgt)).fillna(0.0); val = (1 + r).cumprod() * pd.Series(tgt); L = val.clip(lower=0).sum(axis=1); S = -val.clip(upper=0).sum(axis=1)
        if hedge is not None: hr = B[hedge].get(ym + 1, 0.0); S = pd.Series(E * (1 + hr * np.arange(1, len(days) + 1) / len(days)), index=days)
        if long_only: S = pd.Series(E, index=days)
        pnl_pos = (L - S).diff(); pnl_pos.iloc[0] = L.iloc[0] - S.iloc[0]
        bor = pd.Series(0.0, index=days)
        if not long_only:
            if hedge is None:
                for t in h["short"]: bor += (-val[t]).clip(lower=0) * rates[t] / 252
            else: bor = S * 0.01 / 252
        fees = pd.Series(0.0, index=days); fees.iloc[0] = fee0; eq0 = eq
        for d in days:
            cash_free = max(eq - L.loc[d], 0.0); proceeds = 0.0 if long_only else S.loc[d]
            interest = (cash_free * max(rf.loc[d] - 0.005, 0.0) + max(proceeds - 1e5, 0.0) * max(rf.loc[d] - 0.0125, 0.0)) / 252
            x = pnl_pos.loc[d] - bor.loc[d] - fees.loc[d] + interest; eq += x; rows.append((d, L.loc[d], proceeds, x, bor.loc[d], fees.loc[d], interest, eq))
        pos = {t: val[t].iloc[-1] for t in tgt}
        drows.append(dict(date=ym.to_timestamp(how="end"), sigma=sig, exposure=e, dollars_side=E, n_long=nl, n_short=ns, gap=h["gap"], equity_start=eq0, pnl=eq - eq0, ret=(eq - eq0) / eq0, traded=traded,
                          borrow_rate=float(np.mean(list(rates.values()))) if rates else 0.0, borrow=bor.sum(), fees=fee0, risk_fcst=e * sig, long_ret=L.iloc[-1] / E - 1, short_ret=(S.iloc[-1] / E - 1) if not long_only else np.nan))
    d = pd.DataFrame(rows, columns=["date", "long_value", "short_value", "pnl", "borrow", "fees", "interest", "equity"]).set_index("date")
    d["ret"] = d["pnl"] / d["equity"].shift(1).fillna(capital); d["margin"] = 0.5 * (d["long_value"] + d["short_value"]); d["margin_pct"] = d["margin"] / d["equity"]
    d["margin_stress_pct"] = 0.75 * (d["long_value"] + d["short_value"]) / d["equity"]; d["maint_pct"] = (0.25 * d["long_value"] + 0.30 * d["short_value"]) / d["equity"]
    return d, pd.DataFrame(drows).set_index("date")

def stats(d, m, capital=CAPITAL):
    r = d["ret"]; n = len(r); eq = d["equity"]; dd = eq / eq.cummax() - 1; mr = m["ret"]
    return dict(final_equity=eq.iloc[-1], pnl_usd=eq.iloc[-1] - capital, cagr_pct=((eq.iloc[-1] / capital) ** (252 / n) - 1) * 100, vol_pct=r.std() * np.sqrt(252) * 100,
                sharpe=r.mean() / r.std() * np.sqrt(252), sharpe_monthly=mr.mean() / mr.std() * np.sqrt(12), maxdd_pct=dd.min() * 100, maxdd_usd=(eq - eq.cummax()).min(),
                worst_month_pct=mr.min() * 100, best_month_pct=mr.max() * 100, hit_pct=(mr > 0).mean() * 100, avg_side_pct=m["exposure"].mean() * 100, max_side_pct=m["exposure"].max() * 100,
                avg_margin_pct=d["margin_pct"].mean() * 100, peak_margin_pct=d["margin_pct"].max() * 100, peak_margin_stress_pct=d["margin_stress_pct"].max() * 100,
                fees_usd=d["fees"].sum(), borrow_usd=d["borrow"].sum(), interest_usd=d["interest"].sum(), borrow_rate_pct=m["borrow_rate"].mean() * 100, n_short=m[m.n_short > 0]["n_short"].mean(),
                peak_maint_pct=d["maint_pct"].max() * 100, call_days=int((d["maint_pct"] > 1).sum()), flat_months=int((m["exposure"] == 0).sum()))

t0 = time.time(); H_full = hold_all(short_floor=0.0); H_25 = hold_all(short_floor=2.5e7); H_alt = hold_all(model=ALT) if has_alt else None
variants = {"Headline: $10m short leg, 10% vol target": dict()}
if has_alt: variants[f"{ALT_NAME} forecasts, 10% vol"] = dict(H=H_alt)
variants.update({"Full short leg (>= $1m), tiered borrow, 10% vol": dict(H=H_full), "$25m short leg, 10% vol": dict(H=H_25), "Long leg only, 10% vol": dict(long_only=True), "Long leg, short PGJ, 10% vol": dict(hedge="PGJ"),
                 "Headline, 15% vol target": dict(target=0.15), "Headline, 20% vol target": dict(target=0.20), "Headline, fixed 50% a side": dict(fixed=0.50), "Headline, fixed 80% a side": dict(fixed=0.80),
                 "Headline, 60-day vol only": dict(vol=V60), "Headline, harsh borrow": dict(borrow="harsh"), "Headline, flat 3% borrow": dict(borrow="flat3"), "Headline, flat 8% borrow": dict(borrow="flat8"),
                 "Headline, no borrow cost": dict(borrow="zero"), "Headline, cost 10 bp/side": dict(cost=0.001), "Headline, cost 40 bp/side": dict(cost=0.004)})
runs = {k: simulate(**kw) for k, kw in variants.items()}; S = pd.DataFrame({k: stats(*v) for k, v in runs.items()}).T
print(f"\n[3] Account results, $500,000 start, Jan 2010 - Aug 2026, {COST*1e4:.0f} bp per side, tiered borrow [{time.time()-t0:.0f}s]:")
print(S[["final_equity", "pnl_usd", "cagr_pct", "vol_pct", "sharpe", "sharpe_monthly", "maxdd_pct", "worst_month_pct", "hit_pct", "avg_side_pct", "peak_margin_pct", "call_days", "fees_usd", "borrow_usd", "interest_usd"]].round(2).to_string())
d, m = runs["Headline: $10m short leg, 10% vol target"]; ls = (m["long_ret"] - m["short_ret"]).dropna()
print(f"unit check (1x a side, gross): monthly Sharpe {sharpe_m(ls):+.2f}, {ls.mean()*1200:+.1f}%/yr, long {m['long_ret'].mean()*1200:+.1f}%, short {m['short_ret'].mean()*1200:+.1f}%")
yr = pd.DataFrame({"pnl_usd": d["pnl"].groupby(d.index.year).sum(), "ret_pct": d["ret"].groupby(d.index.year).apply(lambda x: (1 + x).prod() - 1) * 100, "equity_end": d["equity"].groupby(d.index.year).last(),
                   "avg_side_pct": m["exposure"].groupby(m.index.year).mean() * 100, "sigma_pct": m["sigma"].groupby(m.index.year).mean() * 100, "borrow_usd": d["borrow"].groupby(d.index.year).sum(),
                   "peak_margin_pct": d["margin_pct"].groupby(d.index.year).max() * 100, "sharpe": d["ret"].groupby(d.index.year).apply(lambda x: x.mean() / x.std() * np.sqrt(252))})
print("\nby calendar year:"); print(yr.round(1).to_string())
sub = {k: {s: (lambda x: x.mean() / x.std() * np.sqrt(252))(runs[k][0]["ret"].loc[a:b]) for s, (a, b) in SUBS.items()} for k in list(variants)[:6]}; SUBT = pd.DataFrame(sub).T; print("\nsub-period Sharpe:"); print(SUBT.round(2).to_string())

# Monte Carlo
keys = {"Headline": "Headline: $10m short leg, 10% vol target", **({ALT_NAME: f"{ALT_NAME} forecasts, 10% vol"} if has_alt else {}), "Full short leg": "Full short leg (>= $1m), tiered borrow, 10% vol", "Long leg only": "Long leg only, 10% vol", "Long leg, short PGJ": "Long leg, short PGJ, 10% vol"}
M = pd.DataFrame({k: runs[v][1]["ret"] for k, v in keys.items()}); T = len(M); NB = 10000; LB = 12; q = [5, 25, 50, 75, 95]
def bootstrap(M, rng, L=LB, capital=CAPITAL):
    T = len(M); nb = int(np.ceil(T / L)); starts = rng.integers(0, T, (NB, nb)); bidx = ((starts[:, :, None] + np.arange(L)[None, None, :]).reshape(NB, -1) % T)[:, :T]; boot = {}; paths = {}
    for k in M.columns:
        x = M[k].values[bidx]; e = capital * np.cumprod(1 + x, axis=1); pk = np.maximum.accumulate(np.concatenate([np.full((NB, 1), capital), e], axis=1), axis=1)[:, 1:]
        boot[k] = pd.DataFrame({"sharpe": x.mean(1) / x.std(1) * np.sqrt(12), "final_equity": e[:, -1], "cagr_pct": ((e[:, -1] / capital) ** (12 / T) - 1) * 100, "maxdd_pct": (e / pk - 1).min(1) * 100, "eq_1y": e[:, 11], "eq_5y": e[:, min(59, T - 1)]}); paths[k] = e
    return boot, paths
def summ(v): return dict(mean=v.mean(), **{f"p{p:02d}": np.percentile(v, p) for p in q})
def pnl_summary(boot, capital=CAPITAL):
    PL = {}
    for k in boot:
        b = boot[k]; PL[k] = dict(final_p05=np.percentile(b["final_equity"], 5), final_p25=np.percentile(b["final_equity"], 25), final_p50=np.percentile(b["final_equity"], 50), final_p75=np.percentile(b["final_equity"], 75), final_p95=np.percentile(b["final_equity"], 95),
                                  prob_end_below_start=(b["final_equity"] < capital).mean(), cagr_p05=np.percentile(b["cagr_pct"], 5), cagr_p50=np.percentile(b["cagr_pct"], 50), cagr_p95=np.percentile(b["cagr_pct"], 95),
                                  maxdd_p50=np.percentile(b["maxdd_pct"], 50), maxdd_p05=np.percentile(b["maxdd_pct"], 5), maxdd_p01=np.percentile(b["maxdd_pct"], 1), prob_dd_gt20=(b["maxdd_pct"] < -20).mean(), prob_dd_gt30=(b["maxdd_pct"] < -30).mean(),
                                  pnl1y_p05=np.percentile(b["eq_1y"], 5) - capital, pnl1y_p50=np.percentile(b["eq_1y"], 50) - capital, pnl1y_p95=np.percentile(b["eq_1y"], 95) - capital, prob_loss_1y=(b["eq_1y"] < capital).mean(),
                                  pnl5y_p05=np.percentile(b["eq_5y"], 5) - capital, pnl5y_p50=np.percentile(b["eq_5y"], 50) - capital, pnl5y_p95=np.percentile(b["eq_5y"], 95) - capital, prob_loss_5y=(b["eq_5y"] < capital).mean())
    return pd.DataFrame(PL).T
def null_sort(m, n_draw, rng):
    """Random rankings through the same construction, exposure path, costs and borrow."""
    e_path = m["exposure"]; out = np.empty(n_draw)
    for i in range(n_draw):
        LIQ["p_null"] = rng.random(len(LIQ)); Hn = hold_all(col="p_null"); rows = []; pl = ps = set()
        for ym, h in Hn.items():
            if ym not in H: continue
            days = month_days(ym); e = e_path.get(ym.to_timestamp(how="end"), np.nan)
            if len(days) == 0 or pd.isna(e) or e == 0: continue
            sl, ss = set(h["long"]), set(h["short"]); to = ((1 - len(sl & pl) / len(sl)) + (1 - len(ss & ps) / len(ss))) if pl else 2.0; pl, ps = sl, ss
            bor = np.mean([tier(a, BORROW["base"]) for a in h["adv_short"]]) / 12
            rows.append(e * ((leg_paths(h["long"], days).iloc[-1] - 1) - (leg_paths(h["short"], days).iloc[-1] - 1) - bor - COST * 2 * to))
        out[i] = sharpe_m(pd.Series(rows))
    return out
def stale_curve():
    stale = {}
    for s in [0, 1, 2, 3, 4, 6, 12, 24, 60]:
        lag = F[["ticker", "ym", f"p_{MODEL}"]].copy(); lag["ym"] = lag["ym"] + s; LIQ["p_stale"] = LIQ[["ticker", "ym"]].merge(lag, on=["ticker", "ym"], how="left")[f"p_{MODEL}"].fillna(F[f"p_{MODEL}"].median()).values
        rows = [leg_paths(h["long"], month_days(ym)).iloc[-1] - leg_paths(h["short"], month_days(ym)).iloc[-1] for ym, h in hold_all(col="p_stale").items() if len(month_days(ym)) and ym >= pd.Period("2010-01", "M") + s]
        x = pd.Series(rows); stale[s] = dict(months=len(x), gross_sharpe=sharpe_m(x), gross_ret_pct=x.mean() * 1200)
    return pd.DataFrame(stale).T
trials = pd.read_csv(f"{R}/research_quintile_returns.csv"); trial = trials.assign(ls=trials.q5 - trials.q1).groupby("strategy").ls.apply(lambda s: s.mean() / s.std()); V = trial.var(); g_em = 0.5772156649
def dsr(x, nt):
    sr0 = np.sqrt(V) * ((1 - g_em) * norm.ppf(1 - 1 / nt) + g_em * norm.ppf(1 - 1 / (nt * np.e))); sr = x.mean() / x.std(); g3 = skew(x); g4 = kurtosis(x, fisher=False)
    return norm.cdf((sr - sr0) * np.sqrt(len(x) - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)), sr0 * np.sqrt(12)

def monte_carlo(M, runs, keys, d, m, tag, rng, n_null, do_stale):
    boot, paths = bootstrap(M, rng); Bt = pd.DataFrame({k: dict(**summ(boot[k]["sharpe"]), prob_gt0=(boot[k]["sharpe"] > 0).mean()) for k in boot}).T
    Bt.loc["Headline", "prob_beats_longonly"] = (boot["Headline"]["sharpe"] > boot["Long leg only"]["sharpe"]).mean(); Bt.loc["Headline", "prob_beats_full"] = (boot["Headline"]["sharpe"] > boot["Full short leg"]["sharpe"]).mean()
    print(f"\n[{tag}] Block bootstrap ({LB}-month blocks, {NB:,} paths of {len(M)} months) of realised Sharpe:"); print(Bt.round(2).to_string())
    PLT = pnl_summary(boot); print("bootstrap of the P&L path:"); print(PLT.round(3).T.to_string())
    perm = null_sort(m, n_null, rng); obs = sharpe_m(M["Headline"]); nulls = dict(observed=obs, null_mean=perm.mean(), null_sd=perm.std(), null_p95=np.percentile(perm, 95), null_max=perm.max(), p_null=(perm >= obs).mean())
    print(f"random-sort null ({n_null} draws):", {k: round(float(v), 3) for k, v in nulls.items()})
    DS = {k: dict(sharpe=sharpe_m(M[k]), skew=skew(M[k]), kurt=kurtosis(M[k], fisher=False), dsr_13=dsr(M[k], len(trial))[0], dsr_70=dsr(M[k], 70)[0]) for k in M.columns}
    print(f"deflated Sharpe: expected max under the null {dsr(M['Headline'], len(trial))[1]:.2f} ({len(trial)} trials) / {dsr(M['Headline'], 70)[1]:.2f} (70 trials)"); print(pd.DataFrame(DS).T.round(2).to_string())
    extra = {}
    if do_stale: ST = stale_curve(); ST.to_csv(f"{R}/{tag}_stale.csv"); extra["stale"] = ST.to_dict("index"); print("stale forecasts (gross unit long-short):"); print(ST.round(2).to_string())
    d.to_csv(f"{R}/{tag}_daily.csv"); m.to_csv(f"{R}/{tag}_monthly.csv")
    pd.DataFrame({k: runs[v][0]["equity"] for k, v in keys.items()}).to_csv(f"{R}/{tag}_equity_curves.csv")
    pd.DataFrame({f"{k}|{c}": boot[k][c] for k in boot for c in ["sharpe", "final_equity", "maxdd_pct", "eq_1y", "eq_5y"]}).to_csv(f"{R}/{tag}_bootstrap.csv", index=False)
    pd.DataFrame({f"p{p:02d}": np.percentile(paths["Headline"], p, axis=0) for p in q}).to_csv(f"{R}/{tag}_bootstrap_fan.csv", index=False); pd.DataFrame({"null_sharpe": perm}).to_csv(f"{R}/{tag}_null.csv", index=False)
    json.dump(dict(bootstrap_sharpe=Bt.to_dict("index"), bootstrap_pnl=PLT.to_dict("index"), nulls=nulls, deflated=DS, cost=COST, target=TARGET, cap=CAP, cleaning=cstats, **extra), open(f"{R}/{tag}_mc.json", "w"), indent=1, default=float)

S.to_csv(f"{R}/{TAG}_variants.csv"); yr.to_csv(f"{R}/{TAG}_by_year.csv"); SUBT.to_csv(f"{R}/{TAG}_subperiods.csv"); pd.DataFrame({"unit_ls": U, "vol60": V60, "vol": VOL}).to_csv(f"{R}/{TAG}_unit_daily.csv")
monte_carlo(M, runs, keys, d, m, TAG, rng, 500, True)

# the last five years as a fresh account
START5 = "2021-09"; rng5 = np.random.default_rng(5)
v5 = {k: variants[k] for k in variants if k not in ("Headline, 60-day vol only", "Headline, flat 8% borrow", "Headline, cost 10 bp/side")}
runs5 = {k: simulate(**{**kw, "start": START5}) for k, kw in v5.items()}; S5 = pd.DataFrame({k: stats(*v) for k, v in runs5.items()}).T
print(f"\n===== LAST FIVE YEARS: fresh $500,000 account, Sep 2021 - Aug 2026 =====")
print(S5[["final_equity", "pnl_usd", "cagr_pct", "vol_pct", "sharpe", "sharpe_monthly", "maxdd_pct", "worst_month_pct", "hit_pct", "avg_side_pct", "peak_margin_pct", "fees_usd", "borrow_usd", "interest_usd"]].round(2).to_string())
d5, m5 = runs5["Headline: $10m short leg, 10% vol target"]
yr5 = pd.DataFrame({"pnl_usd": d5["pnl"].groupby(d5.index.year).sum(), "ret_pct": d5["ret"].groupby(d5.index.year).apply(lambda x: (1 + x).prod() - 1) * 100, "equity_end": d5["equity"].groupby(d5.index.year).last(),
                    "avg_side_pct": m5["exposure"].groupby(m5.index.year).mean() * 100, "borrow_usd": d5["borrow"].groupby(d5.index.year).sum(), "peak_margin_pct": d5["margin_pct"].groupby(d5.index.year).max() * 100, "sharpe": d5["ret"].groupby(d5.index.year).apply(lambda x: x.mean() / x.std() * np.sqrt(252))})
print("by calendar year:"); print(yr5.round(1).to_string())
keys5 = {k: v for k, v in keys.items() if k != "Long leg, short PGJ"}; M5 = pd.DataFrame({k: runs5[v][1]["ret"] for k, v in keys5.items()})
sy5 = pd.DataFrame({k: runs5[v][0]["ret"].groupby(runs5[v][0].index.year).apply(lambda x: x.mean() / x.std() * np.sqrt(252)) for k, v in keys5.items()}); sy5.loc["all 60 months"] = [S5.loc[v, "sharpe"] for v in keys5.values()]
roll = M["Headline"].rolling(60).apply(sharpe_m); print(f"rolling 60-month Sharpe of the 2010-start account: latest {roll.iloc[-1]:+.2f}, min {roll.min():+.2f}, median {roll.median():+.2f}, max {roll.max():+.2f}")
S5.to_csv(f"{R}/{TAG}_5y_variants.csv"); yr5.to_csv(f"{R}/{TAG}_5y_by_year.csv"); sy5.to_csv(f"{R}/{TAG}_5y_sharpe_by_year.csv")
monte_carlo(M5, runs5, keys5, d5, m5, f"{TAG}_5y", rng5, 300, False)
print(f"\nDONE [{time.time()-t0:.0f}s]")
