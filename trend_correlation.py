"""Correlation of the strategy's monthly returns with trend-following (the SG Trend Index or free proxies for it).

    python3 trend_correlation.py                 # proxies from Yahoo: AQMIX, ASFYX, EBSIX, PQTIX, DBMF, KMLM, CTA and their equal-weight composite
    python3 trend_correlation.py sg_trend.csv    # your own series: a csv with columns date (any month-end or YYYY-MM) and ret (decimal monthly return), e.g. Bloomberg NEIXCTAT

The SG Trend Index history is not published for download (its data is fee-liable on LSEG / Bloomberg), so by default this uses the investable
trend-following funds that track it most closely. Strategy series: the $500k account's monthly return (results/bt500k_rf_monthly.csv) and the
unit long-short before sizing; both are dated by the month in which the return is realised."""
import sys, numpy as np, pandas as pd
from adr_longshort import ROOT
R = f"{ROOT}/results"; pd.set_option("display.width", 200)
m = pd.read_csv(f"{R}/bt500k_rf_monthly.csv", index_col=0, parse_dates=True); m.index = m.index.to_period("M") + 1
S = pd.DataFrame({"account": m.ret, "unit_ls": m.long_ret - m.short_ret})
if len(sys.argv) > 1:
    x = pd.read_csv(sys.argv[1]); x.index = pd.PeriodIndex(pd.to_datetime(x.iloc[:, 0]), freq="M"); T = x.iloc[:, [1]].astype(float); T.columns = ["sg_trend"]; comp = T["sg_trend"]
else:
    import yfinance as yf
    tick = ["AQMIX", "ASFYX", "EBSIX", "PQTIX", "DBMF", "KMLM", "CTA"]; px = yf.download(tick, start="2009-12-01", auto_adjust=True, progress=False)["Close"]
    T = px.resample("ME").last().pct_change(); T.index = T.index.to_period("M"); comp = T[["AQMIX", "ASFYX", "EBSIX", "PQTIX", "DBMF", "KMLM"]].mean(axis=1); T.insert(0, "trend_composite", comp)
X = S.join(T, how="inner").dropna(subset=["account"])
rows = []
for c in T.columns:
    for s in S.columns:
        d = X[[s, c]].dropna()
        if len(d) < 24: continue
        r = d[s].corr(d[c]); rows.append(dict(series=c, strategy=s, months=len(d), corr=r, beta=np.cov(d[s], d[c])[0, 1] / np.var(d[c]), t=r * np.sqrt(len(d) - 2) / np.sqrt(1 - r ** 2)))
out = pd.DataFrame(rows); print("Correlation of monthly returns:"); print(out.pivot(index="series", columns="strategy", values="corr").round(2).join(out[out.strategy == "account"].set_index("series").months).to_string())
c0 = T.columns[0]; d = X[["account", c0]].dropna(); sh = lambda x: x.mean() / x.std() * np.sqrt(12)
print(f"\nAgainst {c0} ({len(d)} months):")
for lab, (a, b) in {"2010-15": ("2010", "2015"), "2016-20": ("2016", "2020"), "2021-26": ("2021", "2026")}.items():
    z = d.loc[a:b]
    if len(z) > 12: print(f"  {lab}: corr {z.account.corr(z[c0]):+.2f} ({len(z)} months)")
dn, up = d[d[c0] < 0], d[d[c0] > 0]; print(f"  strategy in trend-down months {dn.account.mean()*1200:+.1f}%/yr ({len(dn)}), in trend-up months {up.account.mean()*1200:+.1f}%/yr ({len(up)})")
w = d.nsmallest(12, c0); print(f"  worst 12 trend months: trend {w[c0].mean()*100:+.1f}%/month, strategy {w.account.mean()*100:+.2f}%/month, positive in {(w.account > 0).mean():.0%}")
rc = d.account.rolling(36).corr(d[c0]).dropna(); print(f"  rolling 36-month correlation: min {rc.min():+.2f}, median {rc.median():+.2f}, max {rc.max():+.2f}, latest {rc.iloc[-1]:+.2f}")
ta = d[c0] * d.account.std() / d[c0].std(); print(f"  Sharpe: strategy {sh(d.account):.2f}, trend {sh(d[c0]):.2f}, 50/50 at equal volatility {sh(0.5 * d.account + 0.5 * ta):.2f}")
X.to_csv(f"{R}/trend_correlation_series.csv"); out.to_csv(f"{R}/trend_correlation.csv", index=False); print("written results/trend_correlation.csv, results/trend_correlation_series.csv")
