"""Phase 5 follow-ups. (a) AMTD Digital (HKD): when it entered the panel, whether it was ever shorted, what its mid-2022 spike would have done.
(b) HKEX natural experiment: is the bottom-quintile underperformance concentrated in names that cannot be shorted (Miller 1977)?
    python research/p5_hkd_hkshort.py      (after p5_run.py)"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "research"); sys.path.insert(0, ".")
from p5_pipeline import D, log_trial
from adr_longshort import holdings
pd.set_option("display.width", 250)
# ---------- (a) HKD ----------
P = pd.concat([pd.read_csv(f"{D}/gc_ext_prices.csv.gz", parse_dates=["date"]), pd.read_csv(f"{D}/gc_add_prices.csv.gz", parse_dates=["date"])]); h = P[P.ticker == "HKD"].sort_values("date").reset_index(drop=True)
M = pd.read_csv(f"{D}/panel_gc.csv.gz"); m = M[M.ticker == "HKD"]; F = pd.read_csv("research/out/p5_pred_gc.csv"); F["ym"] = pd.PeriodIndex(F.ym, freq="M")
print(f"(a) HKD daily data {h.date.min().date()}..{h.date.max().date()}, {len(h)} rows (needs >= 130 rows in total to enter the panel at all)")
print("    first panel month", m.ym.min(), "| first tradeable month", m[m.tradeable].ym.min(), "| first in-sample (scored) month", m[m.in_sample].ym.min())
print(m[["ym", "ndays", "price", "px_traded", "adv", "adv_mean", "tradeable", "in_sample", "ret_1m", "ret_next"]].head(8).round(3).to_string(index=False))
w = h[(h.date >= "2022-07-10") & (h.date <= "2022-09-30")].copy(); w["ret"] = w.close.pct_change()
print("    July-Sep 2022 path: IPO close %.2f, peak %.2f on %s, 31 Aug %.2f, 30 Sep %.2f" % (w.close.iloc[0], w.close.max(), w.loc[w.close.idxmax(), "date"].date(),
      w[w.date <= "2022-08-31"].close.iloc[-1], w[w.date <= "2022-09-30"].close.iloc[-1]))
book = []
for ym, g in F[F.liquid].groupby("ym"):
    hh = holdings(g)
    if hh and "HKD" in g.ticker.values:
        r = g.set_index("ticker"); side = "short" if "HKD" in hh["short"] else "long" if "HKD" in hh["long"] else "-"
        book.append((str(ym), side, round(r.loc["HKD", "adv"] / 1e6, 2), round(g.p_RF.rank(pct=True)[g.ticker == "HKD"].iloc[0], 2), round(r.loc["HKD", "exret_next"], 3)))
B = pd.DataFrame(book, columns=["formed", "side", "median_adv_$m", "forecast_pctile", "next_month_exret"]); print("    months HKD was in the liquid pool:", len(B), "| shorted:", (B.side == "short").sum(), "| longed:", (B.side == "long").sum())
print(B.to_string(index=False)); B.to_csv("research/out/p5_hkd_book.csv", index=False)
# hypothetical: a short at 2.7% of equity (24% a side / 9 names) opened at the first close after listing, held to 31 Aug
p0 = w.close.iloc[0]; pk = w.close.max(); print(f"    hypothetical 2.7%-of-equity short from the first close: at the peak the position is {pk/p0:.0f}x, i.e. a loss of {0.027*(pk/p0-1)*100:.0f}% of equity")
# ---------- (b) HKEX shortable vs not ----------
X = pd.read_excel(f"{D}/hkex_list.xlsx", header=2); X["ticker"] = X["Stock Code"].astype(int).map(lambda c: f"{c:04d}.HK"); short_ok = set(X[X["Shortsell Eligible"].astype(str).str.upper().str.startswith("Y")].ticker)
rows = []
for kind in ["native", "transfer"]:
    try: F = pd.read_csv(f"research/out/p5_pred_hk_{kind}.csv")
    except FileNotFoundError: continue
    if not len(F): continue
    F["ym"] = pd.PeriodIndex(F.ym, freq="M"); F["shortable"] = F.ticker.isin(short_ok)
    F["q"] = F.groupby("ym").p_RF.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1); F["ex"] = F.exret_next - F.groupby("ym").exret_next.transform("mean")
    F["size3"] = F.groupby("ym").adv.transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["small", "mid", "large"]))
    for sz in ["all", "small", "mid", "large"]:
        G = F if sz == "all" else F[F.size3 == sz]
        for sh in [True, False]:
            g = G[G.shortable == sh]; mq = g.groupby(["ym", "q"]).ex.mean().unstack()
            if mq.empty: continue
            q1 = mq.get(1); q5 = mq.get(5); spread = (q5 - q1).dropna()
            rows.append(dict(model=kind, size=sz, shortable=sh, rows=len(g), q1_vs_month_ew_pct_yr=q1.mean() * 1200 if q1 is not None else np.nan, q5_vs_month_ew_pct_yr=q5.mean() * 1200 if q5 is not None else np.nan,
                             q5_minus_q1_pct_yr=spread.mean() * 1200, t=spread.mean() / spread.std() * np.sqrt(len(spread)) if len(spread) > 2 else np.nan, months=len(spread)))
R = pd.DataFrame(rows); R.to_csv("research/out/p5_hk_shortable.csv", index=False); print("\n(b) HKEX: excess return vs the month's equal-weight cross-section, by forecast quintile and current short-sale eligibility"); print(R.round(2).to_string(index=False))
print(f"    shortable flag is TODAY's list ({len(short_ok)} names); point-in-time history is not published beyond recent weeks")
for kind in R.model.unique():
    a = R[(R.model == kind) & (R["size"] == "all")].set_index("shortable")
    log_trial(f"p5_hk_short_{kind}", f"HKEX Miller test ({kind}): Q1 vs EW, non-shortable vs shortable", f"Q1 {a.loc[False,'q1_vs_month_ew_pct_yr']:.1f}%/yr vs {a.loc[True,'q1_vs_month_ew_pct_yr']:.1f}%/yr", "research/out/p5_hk_shortable.csv")
log_trial("p5_hkd", "AMTD Digital (HKD) entry and book check", f"liquid-pool months {len(B)}, shorted {(B.side=='short').sum()}", "research/out/p5_hkd_book.csv")
