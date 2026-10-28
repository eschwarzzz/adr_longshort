"""Phase 4.1a: Abdi-Ranaldo (2017) effective spread for every ticker-month from daily OHLC, plus the tick floor and commission on the
price as actually traded. Writes research/out/p4_spreads.csv (ticker, ym, ar_spread, tick_half_bp, comm_bp, half_spread_bp, cost_side_bp, sigma_d)."""
import glob, os, numpy as np, pandas as pd
Y = pd.read_csv("data/yf_prices.csv.gz", parse_dates=["date"]).rename(columns=str.lower)[["ticker", "date", "high", "low", "close"]]
K = []
for f in sorted(glob.glob("data/kaggle_patch/stocks/*.csv")):
    t = os.path.basename(f)[:-4]; k = pd.read_csv(f, parse_dates=["Date"]); a = k["Adj Close"] / k.Close
    K.append(pd.DataFrame({"ticker": t, "date": k.Date, "high": k.High * a, "low": k.Low * a, "close": k["Adj Close"]}))
K = pd.concat(K); K = K[~K.ticker.isin(Y.ticker.unique())]; P = pd.concat([Y, K]).dropna().query("close > 0 and high > 0 and low > 0").sort_values(["ticker", "date"])
R = pd.read_csv("research/data/real_prices.csv.gz", parse_dates=["date"])[["ticker", "date", "close_real"]]; P = P.merge(R, on=["ticker", "date"], how="left")
P["c"] = np.log(P.close); P["eta"] = (np.log(P.high) + np.log(P.low)) / 2; P["eta_n"] = P.groupby("ticker").eta.shift(-1)
P["x"] = (P.c - P.eta) * (P.c - P.eta_n); P["r"] = P.groupby("ticker").c.diff(); P["ym"] = P.date.dt.to_period("M")
g = P.groupby(["ticker", "ym"]); S = pd.DataFrame({"ar_s2": g.x.mean() * 4, "sigma_d": g.r.std(), "px_real": g.close_real.last()}).reset_index()
S["ar_spread"] = np.sqrt(S.ar_s2.clip(lower=0)); S["tick_half_bp"] = 0.5 * 0.01 / S.px_real * 1e4; S["comm_bp"] = 0.0035 / S.px_real * 1e4
S["half_spread_bp"] = np.maximum(S.ar_spread / 2 * 1e4, S.tick_half_bp); S["cost_side_bp"] = S.half_spread_bp + S.comm_bp
S[["ticker", "ym", "ar_spread", "sigma_d", "px_real", "tick_half_bp", "comm_bp", "half_spread_bp", "cost_side_bp"]].to_csv("research/out/p4_spreads.csv", index=False)
print(S.cost_side_bp.describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
