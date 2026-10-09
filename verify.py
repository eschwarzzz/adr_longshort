"""Integrity check of the shipped files against the code, in about two minutes:
1. rebuild the characteristics for the last two complete months from the price data, score them with model/rf_2026.joblib, and compare with results/predictions_rf.csv;
2. refit the 2026 forest from results/panel.csv (written by build_panel.py, or here if missing; ranks are computed from the file as written, as in the research) and compare its forecasts with the shipped ones;
3. recompute the quintile long-short Sharpe from the shipped forecasts and compare with results/quintile_stats_liquid_ew.csv.

    python3 verify.py"""
import os, sys, numpy as np, pandas as pd
from adr_longshort import load_prices, load_rf, build_panel, add_target, research_mask, rank_transform, load, predict, annual_refit, holdings, load_predictions, ROOT
R = f"{ROOT}/results"; ok = True
def check(name, cond, detail=""):
    global ok; ok &= bool(cond); print(f"[{'ok' if cond else 'FAIL'}] {name} {detail}")

P = load_prices(); F = load_predictions(); months = sorted(F.ym.unique())[-2:]
M, _ = build_panel(P); _, rfm = load_rf(); M = add_target(M, rfm); M["in_sample"] = research_mask(M); M = rank_transform(M, M.in_sample)
b = load(f"{ROOT}/model/rf_2026.joblib")
for ym in months:
    m = M[(M.ym == ym) & M.in_sample].copy(); m["p_new"] = predict(b, m); x = m.merge(F[F.ym == ym][["ticker", "p_RF"]], on="ticker")
    check(f"forecasts {ym}", len(x) == (F.ym == ym).sum() and (x.p_new - x.p_RF).abs().max() < 1e-9, f"{len(x)} names, max |diff| {(x.p_new - x.p_RF).abs().max():.1e}")
    hp = holdings(m.assign(p_RF=m.p_new)); hr = holdings(F[(F.ym == ym) & F.liquid]); check(f"holdings {ym}", hp["long"] == hr["long"] and hp["short"] == hr["short"], f"{len(hr['long'])} longs, {len(hr['short'])} shorts")
if not os.path.exists(f"{R}/panel.csv"): M.drop(columns=[c for c in M.columns if c.endswith("_r")]).to_csv(f"{R}/panel.csv", index=False)
Mi = pd.read_csv(f"{R}/panel.csv"); Mi["ym"] = pd.PeriodIndex(Mi.ym, freq="M"); Mi = Mi[Mi.in_sample].copy(); Mi = rank_transform(Mi); model, info = annual_refit(Mi, 2026); te = Mi[Mi.year == 2026].copy(); te["p_new"] = predict(model, te); x = te.merge(F[["ticker", "ym", "p_RF"]], on=["ticker", "ym"])
check("2026 refit", (x.p_new - x.p_RF).abs().max() < 1e-9, f"depth {info['max_depth']}, leaf {info['min_samples_leaf']}, {len(x)} rows, max |diff| {(x.p_new - x.p_RF).abs().max():.1e}")
L = F[F.liquid]; rows = []
for ym, g in L.groupby("ym"):
    if len(g) < 25: continue
    g = g.copy(); g["q"] = pd.qcut(g.p_RF.rank(method="first"), 5, labels=False); r = g.groupby("q").exret_next.mean(); rows.append(r[4] - r[0])
ls = pd.Series(rows); sh = ls.mean() / ls.std() * np.sqrt(12); S = pd.read_csv(f"{R}/quintile_stats_liquid_ew.csv", index_col=0)
check("quintile Sharpe", abs(sh - S.loc["RF", "ls_sharpe"]) < 1e-6, f"{sh:.3f} vs shipped {S.loc['RF', 'ls_sharpe']:.3f}")
print("ALL OK" if ok else "SOME CHECKS FAILED"); sys.exit(0 if ok else 1)
