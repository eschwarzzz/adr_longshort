"""This month's positions: prices -> characteristics -> random-forest forecasts -> long and short lists -> dollars a side.

    python3 run_live.py [YYYY-MM] [equity] [--model model/rf_2026.joblib]

Uses the shipped data by default (refresh with fetch_data.py). The month defaults to the last complete month in the price file; equity to 500,000;
the model to model/rf_next.joblib. With no realised history of the unit long-short in hand, sizing uses the prior volatility (40%), i.e. 25% a side."""
import sys, os, numpy as np, pandas as pd
from adr_longshort import load_prices, build_panel, rank_transform, load, predict, holdings, dollars_a_side, weights, ROOT
args = [a for a in sys.argv[1:] if not a.startswith("--")]; model_path = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else f"{ROOT}/model/rf_next.joblib"
P = load_prices(); last_full = (P.date.max().to_period("M") - 1) if P.date.max() < P.date.max().to_period("M").to_timestamp(how="end").normalize() else P.date.max().to_period("M")
ym = pd.Period(args[0], "M") if args else last_full; equity = float(args[1]) if len(args) > 1 else 500_000.0
M, stats = build_panel(P[P.date <= ym.to_timestamp(how="end")]); print("price cleaning:", stats, "| panel", f"{len(M):,}", "stock-months,", M.ticker.nunique(), "names")
M = rank_transform(M); bundle = load(model_path); print("model:", {k: v for k, v in bundle.items() if k != "model"})
month = M[(M.ym == ym) & M.tradeable].copy(); month["p_RF"] = predict(bundle, month)
h = holdings(month)
if h is None: print(f"{ym}: flat (too few names)"); sys.exit()
dollars = dollars_a_side(equity, None)
print(f"\n{ym}: {h['n_liquid']} liquid names, {h['n_short_pool']} above $10m a day; forecast gap {h['forecast_gap']*100:.2f}% a month; ${dollars:,.0f} a side on ${equity:,.0f}")
print("long :", ", ".join(h["long"])); print("short:", ", ".join(h["short"]))
w = pd.Series(weights(h, dollars)).round(0).sort_values(); print(w.to_string())
w.rename("dollars").to_csv(f"{ROOT}/results/positions_{ym}.csv"); print(f"written results/positions_{ym}.csv")
