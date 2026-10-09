"""Annual refits of the random forest on the research panel, 2010-2026, with the research protocol (train 2001..Y-3, validate Y-2..Y-1).

    python3 refit.py [--check]

Writes results/predictions_rf.csv (ticker, ym, liquid, adv, exret_next, p_RF; p_ENS is carried over from the existing file when present),
model/rf_2026.joblib, model/rf_next.joblib, model/rf_hyperparams_by_year.csv and model/feature_importance.csv.
--check compares the new forecasts with the existing results/predictions_rf.csv before overwriting and stops if they differ."""
import os, sys, time, json, numpy as np, pandas as pd
from adr_longshort import annual_refit, fit, predict, rank_transform, FEATS, X_COLS, ROOT
from adr_longshort.rf_model import bundle, save
import sklearn
FIRST, LAST = 2010, 2026; R = f"{ROOT}/results"; MD = f"{ROOT}/model"; check = "--check" in sys.argv
M = pd.read_csv(f"{R}/panel.csv"); M["ym"] = pd.PeriodIndex(M.ym, freq="M"); M = M[M.in_sample].copy(); M = rank_transform(M); M["liquid"] = M.adv >= 1e6       # ranks over the in-sample rows, computed from the file as written
print(f"panel: {len(M):,} stock-months, {M.ticker.nunique()} names, {M.ym.min()}..{M.ym.max()}")
old = pd.read_csv(f"{R}/predictions_rf.csv") if os.path.exists(f"{R}/predictions_rf.csv") else None
if old is not None: old["ym"] = pd.PeriodIndex(old.ym, freq="M")
preds, rows, t0 = [], [], time.time(); final = None
for Y in range(FIRST, LAST + 1):
    te = M[M.year == Y].copy()
    if len(te) == 0: continue
    model, info = annual_refit(M, Y); te["p_RF"] = predict(model, te); preds.append(te)
    diff = np.nan
    if old is not None:
        c = te[["ticker", "ym", "p_RF"]].merge(old[["ticker", "ym", "p_RF"]], on=["ticker", "ym"], suffixes=("", "_old")); diff = float((c.p_RF - c.p_RF_old).abs().max()) if len(c) else np.nan
    rows.append(dict(test_year=Y, train_rows=info["train_rows"], train_names=M[(M.year >= 2001) & (M.year <= Y - 3)].ticker.nunique(), val_rows=info["validate_rows"], max_depth=info["max_depth"], min_samples_leaf=info["min_samples_leaf"],
                     val_mse=info["val_mse"], winsor_lo=info["winsor"][0], winsor_hi=info["winsor"][1], max_abs_diff_vs_previous=diff))
    print(f"test {Y}: depth {info['max_depth']}, leaf {info['min_samples_leaf']}, val MSE {info['val_mse']:.5f}, train {info['train_rows']:,} rows, test {len(te):,}; max |diff| vs existing {diff:.1e} [{time.time()-t0:.0f}s]")
    if Y == LAST: final = (model, info)
H = pd.DataFrame(rows).set_index("test_year")
if check:
    worst = H.max_abs_diff_vs_previous.max(); print(f"check: largest difference against the existing forecasts {worst:.2e}")
    if not worst < 1e-8: print("forecasts differ from the existing file; nothing written"); sys.exit(1)
P = pd.concat(preds)[["ticker", "ym", "liquid", "adv", "exret_next", "p_RF"]]
if old is not None and "p_ENS" in old: P = P.merge(old[["ticker", "ym", "p_ENS"]], on=["ticker", "ym"], how="left")
P.to_csv(f"{R}/predictions_rf.csv", index=False); H.to_csv(f"{MD}/rf_hyperparams_by_year.csv")
save(bundle(final[0], final[1], "2001-2023", "2024-2025", "2026"), f"{MD}/rf_2026.joblib")
tr = M[(M.year >= 2001) & (M.year <= LAST - 2)]; va = M[M.year >= LAST - 1]; nxt, ninfo = fit(tr, va)
save(bundle(nxt, ninfo, f"2001-{LAST-2}", f"{LAST-1}-{LAST}", "from the next refit"), f"{MD}/rf_next.joblib")
print(f"next-cycle model: depth {ninfo['max_depth']}, leaf {ninfo['min_samples_leaf']}, val MSE {ninfo['val_mse']:.5f}, train {ninfo['train_rows']:,} rows")
pd.DataFrame({"rf_2026": final[0].feature_importances_, "rf_next": nxt.feature_importances_}, index=FEATS).sort_values("rf_2026", ascending=False).to_csv(f"{MD}/feature_importance.csv")
json.dump(dict(features=FEATS, model_columns=X_COLS, transform="within-month percentile rank mapped to [-1, 1]; missing -> 0", target="next calendar month's return minus the 1-month T-bill, winsorised at the training sample's 1st/99th percentiles for fitting",
               universe_for_fitting="price >= $1, average daily dollar volume >= $100k at the signal date, serial reverse-splitters (adjusted price ever > $5,000) removed, >= 10 trading days in the target month",
               estimator="sklearn RandomForestRegressor(n_estimators=300, max_features=0.5, random_state=0); max_depth in {2,4,6} and min_samples_leaf in {20,50} chosen by validation MSE each December",
               refit="each December: train 2001..Y-3, validate Y-2..Y-1, use for the twelve months of Y", sklearn_version=sklearn.__version__), open(f"{MD}/feature_spec.json", "w"), indent=1)
print(f"written: results/predictions_rf.csv ({len(P):,} rows), model/rf_2026.joblib, model/rf_next.joblib, model/rf_hyperparams_by_year.csv, model/feature_importance.csv, model/feature_spec.json [{time.time()-t0:.0f}s]")
