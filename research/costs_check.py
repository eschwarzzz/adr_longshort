"""Cost-model realism: high-low spread estimates for held names, size vs ADV and square-root impact by account size,
borrow break-evens and hard-to-borrow share, cash-carry dependence, and a priced trade list for the latest month."""
import sys, os, glob, numpy as np, pandas as pd
sys.path.insert(0, "."); from adr_longshort import load_predictions, holdings, load_rf
pd.set_option("display.width", 250); O = "research/out"
# ---------- OHLC ----------
Y = pd.read_csv("data/yf_prices.csv.gz", parse_dates=["date"]).rename(columns=str.lower)[["ticker", "date", "high", "low", "close", "volume"]]; Y["src"] = "yahoo"
K = []
for f in sorted(glob.glob("data/kaggle_patch/stocks/*.csv")):
    t = os.path.basename(f)[:-4]; k = pd.read_csv(f, parse_dates=["Date"]); K.append(pd.DataFrame({"ticker": t, "date": k.Date, "high": k.High, "low": k.Low, "close": k.Close, "volume": k.Volume, "src": "kaggle"}))
K = pd.concat(K); K = K[~K.ticker.isin(Y.ticker.unique())]; D = pd.concat([Y, K]).dropna(subset=["high", "low", "close"]); D = D[(D.low > 0) & (D.high >= D.low)].sort_values(["ticker", "date"]).reset_index(drop=True)
D["ym"] = D.date.dt.to_period("M"); lh, ll, lc = np.log(D.high), np.log(D.low), np.log(D.close); g = D.groupby("ticker")
eta = (lh + ll) / 2; D["ar"] = 4 * (lc - eta) * (lc - eta.groupby(D.ticker).shift(-1))                                  # Abdi-Ranaldo 2-day estimator of s^2
h2, l2 = lh.groupby(D.ticker).shift(-1), ll.groupby(D.ticker).shift(-1); beta = (lh - ll) ** 2 + (h2 - l2) ** 2; gam = (np.maximum(lh, h2) - np.minimum(ll, l2)) ** 2
k = 3 - 2 * np.sqrt(2); a = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gam / k); D["cs"] = (2 * (np.exp(a) - 1) / (1 + np.exp(a))).clip(lower=0)  # Corwin-Schultz
D["flat"] = (D.high == D.low)
S = D.groupby(["ticker", "ym"]).agg(ar=("ar", lambda x: np.sqrt(max(x.mean(), 0))), cs=("cs", "mean"), flat=("flat", "mean"), src=("src", "first"), px=("close", "last")).reset_index()
# ---------- holdings ----------
F = load_predictions(); L = F[F.liquid]; PAN = pd.read_csv("results/panel.csv"); PAN["ym"] = pd.PeriodIndex(PAN.ym, freq="M")
rows = []; prev = {}
for ym, gg in L.groupby("ym"):
    h = holdings(gg)
    if not h: continue
    for side in ("long", "short"):
        for t in h[side]: rows.append(dict(ym=ym, ticker=t, side=side, n=len(h[side]), new=t not in prev.get(side, set())))
    prev = {"long": set(h["long"]), "short": set(h["short"])}
Hd = pd.DataFrame(rows).merge(L[["ticker", "ym", "adv", "exret_next"]], on=["ticker", "ym"]).merge(S, on=["ticker", "ym"], how="left") \
       .merge(PAN[["ticker", "ym", "vol_1m", "ivol_12m", "age", "mom_12_1", "ret_1m", "high52", "zero_1m", "price"]], on=["ticker", "ym"], how="left")
Hd["spread"] = Hd[["ar", "cs"]].mean(axis=1); Hd["comm_bp"] = (0.0035 / Hd.price.clip(lower=0.5)).clip(upper=0.01) * 1e4      # IBKR tiered $0.0035/sh (1% cap); adjusted price -> understates for later reverse-splitters
Hd["half_spread_bp"] = Hd.spread / 2 * 1e4; Hd["cost_bp"] = Hd.half_spread_bp + Hd.comm_bp
Hd["advb"] = pd.cut(Hd.adv, [0, 3e6, 1e7, 2.5e7, 1e8, 1e13], labels=["1-3m", "3-10m", "10-25m", "25-100m", ">100m"])
q = lambda c: Hd.groupby(c)[["half_spread_bp", "comm_bp", "cost_bp", "flat"]].median().round(1).assign(names=Hd.groupby(c).size())
print("=== 1. effective spread (median of AR & CS, month before trade), per-side bp ===")
print("by leg:\n", q("side").to_string()); print("by ADV bucket:\n", q("advb").to_string())
Hd["yr"] = Hd.ym.dt.year; yy = Hd.pivot_table(index="yr", columns="side", values="cost_bp", aggfunc="median").round(1); print("median cost per side by year (bp):\n", yy.T.to_string())
w = Hd[Hd.new]; print(f"traded-name-weighted mean cost/side (new entries only): long {w[w.side=='long'].cost_bp.mean():.0f} bp, short {w[w.side=='short'].cost_bp.mean():.0f} bp; source kaggle share {(Hd.src=='kaggle').mean():.0%}")
# ---------- 2. size vs ADV, impact ----------
Mo = pd.read_csv("results/bt500k_rf_monthly.csv", parse_dates=["date"]); Mo["ym"] = Mo.date.dt.to_period("M")
Hd = Hd.merge(Mo[["ym", "dollars_side", "equity_start"]], on="ym", how="left"); Hd["pos"] = Hd.dollars_side / Hd.n; Hd["pct_adv"] = Hd.pos / Hd.adv * 100
Hd["frac"] = Hd.pos / Hd.equity_start     # position as share of equity, scales account-size-free
print("\n=== 2. size vs liquidity ===")
for side in ("long", "short"):
    x = Hd[Hd.side == side]; print(f"$500k account, {side}: position/ADV median {x.pct_adv.median():.3f}%, p95 {x.pct_adv.quantile(.95):.3f}%, max {x.pct_adv.max():.2f}%")
cap = []
for A in [5e5, 2e6, 5e6, 1e7, 2.5e7, 5e7]:
    x = Hd.assign(o=Hd.frac * A); x["padv"] = x.o / x.adv; sig = x.vol_1m.fillna(x.vol_1m.median())
    x["imp_bp"] = 0.1 * sig * np.sqrt(x.padv) * 1e4 * 10 ** 0   # square-root law, Y=0.1 (vs commonly ~0.5-1 on daily vol for full-day participation; 0.1 is light)
    x["imp_bp_hi"] = 0.5 * sig * np.sqrt(x.padv) * 1e4
    tr = x[x.new]
    cap.append(dict(account=f"${A/1e6:g}m", short_med_pct_adv=x[x.side == "short"].padv.median() * 100, short_p90_pct_adv=x[x.side == "short"].padv.quantile(.9) * 100,
                    cost_bp_Y01=(tr.cost_bp + tr.imp_bp).mean(), cost_bp_Y05=(tr.cost_bp + tr.imp_bp_hi).mean(), short_cost_bp_Y05=(tr[tr.side == "short"].cost_bp + tr[tr.side == "short"].imp_bp_hi).mean()))
C = pd.DataFrame(cap).set_index("account"); print(C.round(2).to_string())
xs = Hd[Hd.side == "short"]; med = (xs.frac / xs.adv).median()
print("account size at which the median short hits 1/5/10% of ADV: " + ", ".join(f"{p}% -> ${p/100/med/1e6:,.1f}m" for p in (1, 5, 10)))
p90 = (xs.frac / xs.adv).quantile(.9); print("...the 90th-pct short: " + ", ".join(f"{p}% -> ${p/100/p90/1e6:,.1f}m" for p in (1, 5, 10)))
# drag on the account if realistic costs replace 20 bp: traded dollars x (cost - 20bp)
turn = (Mo.traded / Mo.equity_start).fillna(0); yrs = len(Mo) / 12; ann_turn = turn.sum() / yrs
print(f"account turnover {ann_turn:.2f}x equity a year (both sides); each +10 bp per side costs {ann_turn*10/100:.2f}% a year")
for A, r in C.iterrows(): print(f"  {A}: Y=0.1 {r.cost_bp_Y01:.0f} bp -> drag vs 20bp {(r.cost_bp_Y01-20)*ann_turn/100:+.2f}%/yr; Y=0.5 {r.cost_bp_Y05:.0f} bp -> {(r.cost_bp_Y05-20)*ann_turn/100:+.2f}%/yr")
C.to_csv(f"{O}/costs_capacity.csv")
# ---------- 3. borrow ----------
print("\n=== 3. borrow ===")
sh = Hd[Hd.side == "short"].copy(); sh["months_listed"] = np.exp(sh.age); sh["r12"] = (1 + sh.mom_12_1) * (1 + sh.ret_1m) - 1
print(sh[["months_listed", "r12", "price", "adv", "ivol_12m", "high52"]].describe(percentiles=[.1, .25, .5, .75, .9]).round(2).T[["10%", "25%", "50%", "75%", "90%"]].to_string())
sh["htb"] = (sh.months_listed < 24) | (sh.price < 5) | (sh.r12 < -0.7)
sh["contrib"] = -sh.exret_next / sh.n            # unit short-leg contribution
lg = Hd[Hd.side == "long"].groupby("ym").exret_next.mean(); sl = sh.groupby("ym").exret_next.mean()
yrs_s = sh.groupby(sh.ym.dt.year).htb.mean()
print(f"HTB-flagged share of short name-months {sh.htb.mean():.0%} (2010-17 {sh[sh.yr<=2017].htb.mean():.0%}, 2021-26 {sh[sh.yr>=2021].htb.mean():.0%}); share of short-leg gross P&L from them {sh[sh.htb].contrib.sum()/sh.contrib.sum():.0%}")
print(f"  flagged names' avg short return {-sh[sh.htb].exret_next.mean()*1200:.0f}%/yr vs others {-sh[~sh.htb].exret_next.mean()*1200:.0f}%/yr  (break-even borrow for each group)")
ls = (lg - sl).dropna(); tc = 0.002 * 2 * 2 * 0.53      # ~53% monthly turnover per leg, 20 bp per side, both legs
net0 = ls - tc; mu, sd = net0.mean() * 12, net0.std() * np.sqrt(12)
print(f"unit $3m long-short: gross {ls.mean()*1200:.1f}%/yr, after 20bp trading {mu*100:.1f}%/yr, vol {sd*100:.1f}%. Flat borrow on the short leg for net Sharpe 0: {mu*100:.0f}%/yr; for 0.5: {(mu-0.5*sd)*100:.0f}%/yr; for 1.0: {(mu-1.0*sd)*100:.0f}%/yr")
mm = Mo[Mo.n_short > 0]; print(f"backtest charged average borrow {mm.borrow_rate.mean()*100:.1f}%/yr (2021-26 {mm[mm.date>='2021'].borrow_rate.mean()*100:.1f}%)")
# ---------- 4. cash carry ----------
print("\n=== 4. cash carry ===")
Dd = pd.read_csv("results/bt500k_rf_daily.csv", parse_dates=["date"]).set_index("date"); rf, _ = load_rf(); rfd = rf.reindex(Dd.index).ffill().bfill()
eqp = Dd.equity.shift(1).fillna(500000); free = (eqp - Dd.long_value).clip(lower=0); S_ = Dd.short_value
i_cash = free * (rfd - 0.005).clip(lower=0) / 252; i_short = (S_ - 1e5).clip(lower=0) * (rfd - 0.0125).clip(lower=0) / 252
print(f"reconstructed interest {(i_cash+i_short).sum()/1e3:.0f}k vs file {Dd.interest.sum()/1e3:.0f}k (cash {i_cash.sum()/1e3:.0f}k, short rebate {i_short.sum()/1e3:.0f}k)")
yrr = pd.DataFrame({"pnl_k": Dd.pnl.groupby(Dd.index.year).sum() / 1e3, "interest_k": Dd.interest.groupby(Dd.index.year).sum() / 1e3}); yrr["carry_share"] = yrr.interest_k / yrr.pnl_k
print(yrr.round(1).T.to_string())
def final(extra_daily):
    r = (Dd.pnl - extra_daily) / eqp; return 500000 * (1 + r).prod(), r.mean() / r.std() * np.sqrt(252)
for lab, x in [("headline", 0 * i_cash), ("no short rebate", i_short), ("short rebate 0, cash at bill-1%", i_short + free * 0.005 / 252 * (rfd > 0.005)), ("no interest at all", Dd.interest)]:
    fe, s = final(x); print(f"  {lab:34s} final ${fe/1e6:.2f}m  Sharpe {s:.2f}")
# ---------- 5. latest month's trade list ----------
print("\n=== 5. trade list, 2026-09 (run_live.py, $500k, 25% a side) ===")
pos = pd.read_csv("results/positions_2026-09.csv", index_col=0).squeeze(); last = D[D.ym == pd.Period("2026-09", "M")]
info = last.groupby("ticker").agg(px=("close", "last"), adv=("volume", lambda v: np.nan)).px.to_frame()
adv = last.assign(dv=last.close * last.volume).groupby("ticker").dv.mean(); spr = S[S.ym == pd.Period("2026-09", "M")].set_index("ticker")[["ar", "cs"]].mean(axis=1)
T = pd.DataFrame({"dollars": pos, "px": info.px, "shares": (pos / info.px).round(0), "adv_$m": adv / 1e6, "pct_adv": pos.abs() / adv * 100, "spread_bp": spr * 1e4}).reindex(pos.index)
T["cost_side_bp"] = T.spread_bp / 2 + 0.0035 / T.px * 1e4; print(T.round(2).sort_values("dollars").to_string())
print(f"short leg: median spread {T[T.dollars<0].spread_bp.median():.0f} bp, median px ${T[T.dollars<0].px.median():.2f}; long leg median spread {T[T.dollars>0].spread_bp.median():.0f} bp")
T.to_csv(f"{O}/costs_tradelist_2026-09.csv"); Hd.to_csv(f"{O}/costs_held_names.csv", index=False)
