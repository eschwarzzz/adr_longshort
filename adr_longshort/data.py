"""Loaders for the files under data/ and results/. Paths are relative to the repository root, so scripts run from anywhere."""
import os, glob, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = f"{ROOT}/data"; R = f"{ROOT}/results"; M = f"{ROOT}/model"

def load_prices(start="1999-01-01"):
    """Daily adjusted closes and volumes: the Yahoo file (currently listed names, with the SOE extensions) plus the Kaggle archive for delisted names.
    Columns ticker, date, close, volume. A ticker present in both sources is taken from Yahoo."""
    Y = pd.read_csv(f"{D}/yf_prices.csv.gz", parse_dates=["date"]).rename(columns=str.lower)[["ticker", "date", "close", "volume"]]
    K = []
    for f in sorted(glob.glob(f"{D}/kaggle_patch/stocks/*.csv")):
        t = os.path.basename(f)[:-4]; k = pd.read_csv(f, parse_dates=["Date"]); K.append(pd.DataFrame({"ticker": t, "date": k["Date"], "close": k["Adj Close"], "volume": k["Volume"]}))
    K = pd.concat(K) if K else pd.DataFrame(columns=Y.columns); K = K[~K.ticker.isin(Y.ticker.unique())]
    P = pd.concat([Y, K]).dropna(subset=["close"]).sort_values(["ticker", "date"]); P = P[(P.close > 0) & (P.date >= start)]
    return P.reset_index(drop=True)

def load_rf():
    """1-month T-bill from FRED (DTB4WK, percent per year). Returns (daily series indexed by date, monthly mean per Period[M] as a decimal per month)."""
    rf = pd.read_csv(f"{D}/fred_DTB4WK.csv", na_values=".", parse_dates=["observation_date"]).dropna().set_index("observation_date")["DTB4WK"] / 100
    return rf, rf.groupby(rf.index.to_period("M")).mean() / 12

def load_benchmarks():
    """Monthly total returns of the China ETFs and SPY, indexed by Period[M]."""
    B = pd.read_csv(f"{D}/benchmarks_monthly.csv", index_col=0); B.index = pd.PeriodIndex(B.index, freq="M"); return B

def load_predictions(path=None):
    """The forecasts: ticker, ym, liquid flag, adv, next-month excess return (research only), p_RF and p_ENS."""
    P = pd.read_csv(path or f"{R}/predictions_rf.csv"); P["ym"] = pd.PeriodIndex(P.ym, freq="M"); return P
