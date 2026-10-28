"""Phase 3 construction library on the corrected baseline.
A "book" each month = weighted short names + a long side that is either weighted stock names or a hedge return series
(ETF or universe basket) at a hedge ratio. Unit book (monthly, $1 a side) and a $500k account (daily, same rules as
backtest.py: vol-target sizing on the book's own unit series, 20 bp/side stocks, 2 bp/side ETF, tiered borrow per short name,
interest on free cash at bill-0.5% and on short proceeds above $100k at bill-1.25%)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "research")
from adr_longshort import holdings, sizing_vol
from account import load as load_account
import os; PRED = os.environ.get("P3_PRED", "research/out/p0_pred_corrected.csv")
A = load_account(PRED)                                     # backtest namespace: RET (cleaned daily returns), DAYS, month_days, rf, BORROW, tier, LIQ
RET, DAYS, rf = A.RET, A.DAYS, A.rf
LIQ = A.LIQ.copy(); LIQ["exret_next"] = LIQ.exret_next
ETF = pd.read_csv("research/data/p3/etf_daily.csv", index_col=0, parse_dates=True).pct_change().reindex(DAYS)
COST_STK, COST_ETF = 0.002, 0.0002
def tier(adv, sched): return A.tier(adv, A.BORROW[sched])
MONTHS = sorted(LIQ.ym.unique())

def month_groups():
    return {ym: g for ym, g in LIQ.groupby("ym")}
G = month_groups()

def basket_daily(names, w, days):
    """Buy-and-hold daily returns of a weighted basket over `days` (weights at the start), and per-name value paths."""
    r = RET.reindex(index=days, columns=names).fillna(0.0); v = (1 + r).cumprod() * np.asarray(w)
    tot = v.sum(axis=1); prev = np.r_[np.sum(w), tot.values[:-1]]; return pd.Series(tot.values / prev - 1, index=days), v

def trailing_beta(names, w, hedge_ret, end, n=252, lo=0.5, hi=1.5):
    """Beta of today's short basket (fixed weights, its names' past daily returns) on a hedge series, trailing n days to `end`."""
    d = DAYS[DAYS <= end][-n:]; r = RET.reindex(index=d, columns=names)
    b = (r.fillna(0.0) * np.asarray(w)).sum(axis=1) / np.sum(w); h = hedge_ret.reindex(d)
    ok = h.notna() & r.notna().any(axis=1)
    if ok.sum() < 60: return 1.0
    x, y = h[ok], b[ok]; return float(np.clip(np.cov(y, x)[0, 1] / x.var(), lo, hi))

def universe_basket(g, kind):
    """Liquid names of the month, weights adv (VW) or equal (EW)."""
    names = list(g.ticker); w = (g.adv.values / g.adv.sum()) if kind == "VW" else np.full(len(g), 1 / len(g)); return names, w

def inv_vol_weights(names, end, cap_mult=2.0, cap_abs=0.15, n=60):
    d = DAYS[DAYS <= end][-n:]; v = RET.reindex(index=d, columns=names).std().reindex(names)
    v = v.fillna(v.median() if v.notna().any() else 0.05).clip(lower=1e-4); w = (1 / v).values; w = w / w.sum()
    cap = max(min(cap_mult / len(names), cap_abs), 1 / len(names))      # 15% absolute cap is infeasible below 7 names: fall back to equal weight there
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any(): break
        ex = (w[over] - cap).sum(); w[over] = cap; w[~over] += ex * w[~over] / w[~over].sum()
    return w

def make_holdings(score_adj=None, avail=None):
    """Monthly long/short lists. score_adj(g) -> expected monthly borrow per short candidate, ADDED to p_RF for the short ranking (borrow-aware);
    avail: set of (ticker, ym) unavailable to short -> skipped, next-ranked name taken."""
    H = {}
    for ym, g in G.items():
        h = holdings(g, "p_RF")
        if h is None: continue
        if score_adj is not None or avail is not None:
            gl = g[g.tradeable] if "tradeable" in g else g; gs = gl[gl.adv >= 3e6].copy()
            gs["score"] = gs.p_RF + (score_adj(gs) if score_adj is not None else 0.0)        # short profit = -forecast - borrow, so expensive names rank as less attractive shorts
            if avail is not None: gs = gs[[(t, ym) not in avail for t in gs.ticker]]
            k = max(len(gs) // 5, 5) if len(gs) >= 10 else 0
            if k == 0: continue
            sh = gs.sort_values("score").head(k); h = dict(h, short=list(sh.ticker), adv_short=sh.adv.values)
        H[ym] = h
    return H

def build_book(H, long_kind="stocks", hedge=None, beta_match=False, short_w="equal", ratio=1.0):
    """Per month: short names+weights+rates, long side spec. hedge in {PGJ,KWEB,FXI,MCHI,VW,EW} when long_kind=='hedge'."""
    book = {}
    for ym, h in H.items():
        end = ym.to_timestamp(how="end"); sn = h["short"]
        sw = inv_vol_weights(sn, end) if short_w == "invvol" else np.full(len(sn), 1 / len(sn))
        rates = {"base": [tier(a, "base") for a in h["adv_short"]], "harsh": [tier(a, "harsh") for a in h["adv_short"]]}
        e = dict(short=sn, sw=sw, rates=rates)
        if long_kind == "stocks": e.update(long=h["long"], lw=np.full(len(h["long"]), 1 / len(h["long"])), hedge=None, hr=0.0)
        elif long_kind == "none": e.update(long=[], lw=np.array([]), hedge=None, hr=0.0)
        else:
            if hedge in ("VW", "EW"):
                names, w = universe_basket(G[ym], hedge); e.update(long=names, lw=w, hedge=None)
                hr = trailing_beta(sn, sw, basket_hist(names, w, end), end) if beta_match else ratio; e["lw"] = w * hr; e["hr"] = hr
            else:
                if ETF[hedge].loc[:end].dropna().empty or ETF[hedge].loc[end:].dropna().empty: continue
                hr = trailing_beta(sn, sw, ETF[hedge], end) if beta_match else ratio; e.update(long=[], lw=np.array([]), hedge=hedge, hr=hr)
        book[ym] = e
    return book

def basket_hist(names, w, end, n=252):
    d = DAYS[DAYS <= end][-n:]; r = RET.reindex(index=d, columns=names).fillna(0.0); return (r * w).sum(axis=1) / np.sum(w)

def unit_series(book):
    """Daily unit ($1 a side, hedge at hr) return series and monthly leg table (gross long/hedge, short, costs)."""
    out = []; rows = []; prev_l = {}; prev_s = {}; prev_h = (None, 0.0)
    for ym, e in book.items():
        days = A.month_days(ym)
        if len(days) == 0: continue
        sr_, sv = basket_daily(e["short"], e["sw"], days); Sp = (1 + sr_).cumprod()
        if e["hedge"]: hr_ = ETF[e["hedge"]].reindex(days).fillna(0.0); Lp = (1 + hr_).cumprod() * e["hr"]; Lb = e["hr"]
        elif len(e["long"]): lb_, lv = basket_daily(e["long"], e["lw"], days); Lb = float(np.sum(e["lw"])); Lp = (1 + lb_).cumprod() * Lb
        else: Lp = pd.Series(0.0, index=days); Lb = 0.0
        v = 1 + (Lp - Lb) - (Sp - 1); u = v.pct_change(); u.iloc[0] = v.iloc[0] - 1; out.append(u)     # backtest.py's unit definition: value of $1-a-side long minus short, plus 1
        lr_ = Lp / Lb - 1 if Lb else Lp * 0
        # turnover cost: sum |w_new - w_old| per side
        nl = dict(zip(e["long"], e["lw"])); ns = dict(zip(e["short"], e["sw"]))
        tl = sum(abs(nl.get(t, 0) - prev_l.get(t, 0)) for t in set(nl) | set(prev_l)); ts = sum(abs(ns.get(t, 0) - prev_s.get(t, 0)) for t in set(ns) | set(prev_s))
        th = abs(e["hr"] - prev_h[1]) if e["hedge"] == prev_h[0] else e["hr"] + prev_h[1] if e["hedge"] else (prev_h[1] if prev_h[0] else 0.0)
        cost = COST_STK * (tl + ts) + COST_ETF * th
        bb = float(np.dot(e["sw"], e["rates"]["base"])) / 12; bh = float(np.dot(e["sw"], e["rates"]["harsh"])) / 12
        L = (Lp.iloc[-1] - Lb) if Lb else 0.0; S = Sp.iloc[-1] - 1
        rows.append(dict(ym=ym, L=L, S=S, gross=L - S, cost=cost, b_base=bb, b_harsh=bh, hr=e["hr"], n_short=len(e["short"]), max_w=float(np.max(e["sw"]))))
        prev_l, prev_s, prev_h = nl, ns, (e["hedge"], e["hr"])
    q = pd.DataFrame(rows).set_index("ym"); q["net_base"] = q.gross - q.cost - q.b_base; q["net_harsh"] = q.gross - q.cost - q.b_harsh
    return pd.concat(out), q

def simulate(book, borrow="base", capital=500_000.0, target=0.10, cap=0.80, start="2010-01", end="2026-08"):
    """$500k account on a book. Sizing: dollars a side = min(cap, target/sigma) x equity, sigma from the book's own unit series."""
    U, _ = unit_series(book); vol = sizing_vol(U); eq = capital; pos = {}; rows = []; mrows = []; contrib = []
    for ym in pd.period_range(pd.Period(start, "M") - 1, pd.Period(end, "M") - 1, freq="M"):
        days = A.month_days(ym)
        if len(days) == 0: continue
        if ym not in book:
            traded = sum(abs(v) for v in pos.values()); fee = sum(abs(v) * (COST_ETF if k.startswith("ETF:") else COST_STK) for k, v in pos.items()); eq -= fee; pos = {}; eq0 = eq
            for d in days:
                it = eq * max(rf.loc[d] - 0.005, 0.0) / 252; eq += it; rows.append((d, 0.0, 0.0, it, 0.0, 0.0, it, eq))
            mrows.append(dict(date=ym.to_timestamp(how="end"), exposure=0.0, ret=(eq - eq0) / eq0, dollars=0.0, n_short=0)); continue
        e = book[ym]; sig = vol.loc[:ym.to_timestamp(how="end")].dropna(); sig = sig.iloc[-1] if len(sig) else 0.40
        x = min(cap, target / sig); E = x * eq
        tgt = {t: -E * w for t, w in zip(e["short"], e["sw"])}
        if e["hedge"]: tgt["ETF:" + e["hedge"]] = E * e["hr"]
        else: tgt.update({t: tgt.get(t, 0.0) + E * w for t, w in zip(e["long"], e["lw"])})
        fee0 = sum(abs(tgt.get(k, 0.0) - pos.get(k, 0.0)) * (COST_ETF if k.startswith("ETF:") else COST_STK) for k in set(tgt) | set(pos))
        names = [k for k in tgt if not k.startswith("ETF:")]; r = RET.reindex(index=days, columns=names).fillna(0.0)
        if e["hedge"]: r["ETF:" + e["hedge"]] = ETF[e["hedge"]].reindex(days).fillna(0.0)
        val = (1 + r).cumprod() * pd.Series(tgt)[r.columns]; Lv = val.clip(lower=0).sum(axis=1); Sv = -val.clip(upper=0).sum(axis=1)
        pnl = (val.diff().sum(axis=1)); pnl.iloc[0] = (val.iloc[0] - pd.Series(tgt)[r.columns]).sum()
        rates = dict(zip(e["short"], e["rates"][borrow])); bor = pd.Series(0.0, index=days)
        for t in e["short"]:
            if t in val: bor += (-val[t]).clip(lower=0) * rates[t] / 252
        eq0 = eq
        for i, d in enumerate(days):
            cash_free = max(eq - Lv.loc[d], 0.0); it = (cash_free * max(rf.loc[d] - 0.005, 0.0) + max(Sv.loc[d] - 1e5, 0.0) * max(rf.loc[d] - 0.0125, 0.0)) / 252
            xx = pnl.loc[d] - bor.loc[d] - (fee0 if i == 0 else 0.0) + it; eq += xx; rows.append((d, Lv.loc[d], Sv.loc[d], xx, bor.loc[d], fee0 if i == 0 else 0.0, it, eq))
        for t in e["short"]:
            if t in val: contrib.append(dict(ym=ym, ticker=t, loss_pct=float((val[t].iloc[-1] - tgt[t]) / eq0 * 100), w=float(-tgt[t] / E)))
        pos = {k: val[k].iloc[-1] for k in val.columns}
        mrows.append(dict(date=ym.to_timestamp(how="end"), exposure=x, ret=(eq - eq0) / eq0, dollars=E, n_short=len(e["short"]), max_short=float(np.max(e["sw"]) * E), long_val=float(Lv.iloc[0]), eq0=eq0, sigma=sig))
    d = pd.DataFrame(rows, columns=["date", "long_value", "short_value", "pnl", "borrow", "fees", "interest", "equity"]).set_index("date"); d["ret"] = d.pnl / d.equity.shift(1).fillna(capital)
    d["maint_pct"] = (0.25 * d.long_value + 0.30 * d.short_value) / d.equity
    return d, pd.DataFrame(mrows).set_index("date"), pd.DataFrame(contrib)

PGJm = None
def acct_stats(d, m, capital=500_000.0):
    r = d.ret; eq = d.equity; dd = eq / eq.cummax() - 1; mr = m.ret
    pg = ETF["PGJ"].reindex(d.index).fillna(0.0); pgm = (1 + pg).groupby(d.index.to_period("M")).prod() - 1; mm = (1 + r).groupby(d.index.to_period("M")).prod() - 1
    b = np.cov(mm, pgm.reindex(mm.index))[0, 1] / pgm.reindex(mm.index).var()
    s = lambda x: x.mean() / x.std() * np.sqrt(252)
    return dict(final_equity=eq.iloc[-1], cagr_pct=((eq.iloc[-1] / capital) ** (252 / len(r)) - 1) * 100, vol_pct=r.std() * np.sqrt(252) * 100, sharpe=s(r), sharpe_monthly=mr.mean() / mr.std() * np.sqrt(12),
                maxdd_pct=dd.min() * 100, worst_month_pct=mr.min() * 100, worst_day_pct=r.min() * 100, beta_pgj_m=b, avg_side_pct=m.exposure.mean() * 100, fees_usd=d.fees.sum(), borrow_usd=d.borrow.sum(), interest_usd=d.interest.sum(),
                sharpe_10_15=s(r.loc["2010":"2015"]), sharpe_16_20=s(r.loc["2016":"2020"]), sharpe_21_26=s(r.loc["2021":]), call_days=int((d.maint_pct > 1).sum()))

def unit_stats(q, U=None):
    s = lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else np.nan
    c = (1 + q.net_base).cumprod()
    out = dict(months=len(q), gross_ret=q.gross.mean() * 1200, gross_sharpe=s(q.gross), net_base_ret=q.net_base.mean() * 1200, net_base_sharpe=s(q.net_base), net_harsh_sharpe=s(q.net_harsh), vol=q.gross.std() * np.sqrt(12) * 100,
               maxdd=(c / c.cummax() - 1).min() * 100, worst_month=q.net_base.min() * 100, cost_pa=q.cost.mean() * 1200, borrow_pa=q.b_base.mean() * 1200, hedge_ratio=q.hr.mean(),
               net_10_15=s(q.net_base.loc["2010":"2015"]), net_16_20=s(q.net_base.loc["2016":"2020"]), net_21_26=s(q.net_base.loc["2021":]))
    return out
