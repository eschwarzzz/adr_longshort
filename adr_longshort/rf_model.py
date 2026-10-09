"""The random-forest forecaster: fit with the research protocol, or load a saved model and score a month of features.

Protocol: each December, for the coming test year Y, train on 2001..Y-3 and pick the hyperparameters on Y-2..Y-1 by validation MSE; the chosen
forest scores every month of Y. Target for fitting = next-month excess return winsorised at the training sample's 1st and 99th percentiles.
Inputs = the 16 rank-transformed characteristics in features.X_COLS, in that order."""
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestRegressor
from .features import X_COLS
GRID = [(d, l) for d in (2, 4, 6) for l in (20, 50)]           # max_depth x min_samples_leaf
N_TREES = 300; MAX_FEATURES = 0.5; SEED = 0

def fit(train, validate):
    """train / validate: panel rows with X_COLS and exret_next. Returns (model, info)."""
    lo, hi = train.exret_next.quantile([0.01, 0.99]); ytr = train.exret_next.clip(lo, hi).values; yva = validate.exret_next.clip(lo, hi).values
    cands = {k: RandomForestRegressor(n_estimators=N_TREES, max_depth=k[0], min_samples_leaf=k[1], max_features=MAX_FEATURES, n_jobs=-1, random_state=SEED).fit(train[X_COLS].values, ytr) for k in GRID}
    mse = {k: float(((yva - m.predict(validate[X_COLS].values)) ** 2).mean()) for k, m in cands.items()}; k = min(mse, key=mse.get)
    return cands[k], dict(max_depth=k[0], min_samples_leaf=k[1], val_mse=mse[k], val_mse_all={f"{a}/{b}": v for (a, b), v in mse.items()}, winsor=(float(lo), float(hi)), train_rows=len(train), validate_rows=len(validate))

def annual_refit(M, test_year):
    """Fit the model the protocol uses for test_year. M: research panel rows (research_mask applied, ranked) with year, X_COLS, exret_next."""
    tr = M[(M.year >= 2001) & (M.year <= test_year - 3)]; va = M[(M.year >= test_year - 2) & (M.year <= test_year - 1)]; return fit(tr, va)

def bundle(model, info, train, validate, test):
    return dict(model=model, features=X_COLS, train=train, validate=validate, test=test, max_depth=info["max_depth"], min_samples_leaf=info["min_samples_leaf"], winsor=info["winsor"])

def save(b, path): joblib.dump(b, path)

def load(path):
    """Saved bundle: dict(model=RandomForestRegressor, features=X_COLS, train, validate, test, max_depth, min_samples_leaf, winsor)."""
    b = joblib.load(path); assert list(b["features"]) == X_COLS, "feature order mismatch"; return b

def predict(bundle_or_model, M):
    """Forecast of next month's excess return (decimal) for each row of M (needs X_COLS, i.e. rank-transformed features)."""
    m = bundle_or_model["model"] if isinstance(bundle_or_model, dict) else bundle_or_model
    return pd.Series(m.predict(M[X_COLS].values), index=M.index, name="p_RF")
