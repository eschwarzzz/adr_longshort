"""Two universe filters that use information from after the signal date:
(a) research_mask needs >= 10 trading days of data NEXT month (names halted or delisted next month are removed from this month's pool);
(b) build_panel drops every name whose adjusted price EVER exceeds $5,000 (serial reverse-splitters), for its whole history."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "."); from adr_longshort import load_prices
M = pd.read_csv("results/panel.csv"); M["ym"] = pd.PeriodIndex(M.ym, freq="M")
pool = M[M.tradeable & (M.adv >= 1e6) & (M.ym >= "2009-12") & (M.ym <= "2026-07")]
ex = pool[~pool.in_sample]
print(f"(a) liquid tradeable rows 2010-26: {len(pool):,}; excluded by the next-month filter: {len(ex)} ({len(ex)/len(pool):.1%}), of which >= $3m a day: {(ex.adv>=3e6).sum()}")
print("    reason: no next-month data at all:", int(ex.ndays_next.isna().sum()), "| < 10 days next month:", int((ex.ndays_next < 10).sum()))
print("    next-month return where a partial month exists: mean %.1f%%, median %.1f%%" % (ex.ret_next.mean() * 100, ex.ret_next.median() * 100))
print("    excluded per year:", ex.groupby(ex.ym.dt.year).size().to_dict())
print(ex.sort_values("ym")[["ticker", "ym", "adv", "price", "ret_1m", "ndays_next", "ret_next"]].assign(adv=lambda d: (d.adv / 1e6).round(1)).to_string(index=False))
P = load_prices(); P["ym"] = P.date.dt.to_period("M")
mx = P.groupby("ticker").close.max(); big = mx[mx > 5000]
inpanel = set(M.ticker)
print(f"\n(b) names with adjusted close ever > $5,000: {len(big)}; absent from panel: {sorted(set(big.index) - inpanel)}")
for t in sorted(set(big.index) - inpanel):
    g = P[P.ticker == t].copy(); g["dv"] = g.close * g.volume; mm = g.groupby("ym").agg(px=("close", "last"), adv=("dv", "mean"), n=("close", "size"))
    liq = mm[(mm.adv >= 1e6) & (mm.px >= 1) & (mm.n >= 15) & (mm.index >= "2009-12")]
    first, last = g.close.iloc[0], g.close.iloc[-1]
    print(f"  {t}: {g.date.min().date()}..{g.date.max().date()}, adj price {first:,.0f} -> {last:,.2f} ({last/first-1:+.1%}), months liquid (>=$1m/day, >=$1) since 2010: {len(liq)}")
