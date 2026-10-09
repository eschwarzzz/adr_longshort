"""Unit tests on synthetic data (fast), plus a consistency test against the shipped forecasts when the data files are present.

    python3 -m pytest -q"""
import os, numpy as np, pandas as pd, pytest
from adr_longshort import clean, build_panel, rank_transform, holdings, sizing_vol, dollars_a_side, weights, FEATS, X_COLS, ROOT, load, predict, load_predictions

def synthetic_prices(n_names=30, n_days=800, seed=0):
    rng = np.random.default_rng(seed); days = pd.bdate_range("2020-01-01", periods=n_days); out = []
    for i in range(n_names):
        r = rng.normal(0.0003, 0.02 + 0.03 * (i / n_names), n_days); px = 10 * np.exp(np.cumsum(r)); vol = rng.integers(1e5, 1e6 * (1 + i), n_days) * (i + 1)
        out.append(pd.DataFrame({"ticker": f"T{i:02d}", "date": days, "close": px, "volume": vol.astype(float)}))
    return pd.concat(out, ignore_index=True)

def test_clean_removes_isolated_print_and_pair():
    days = pd.bdate_range("2021-01-01", periods=12); px = [10, 10.1, 40, 10.2, 10.3, 10.1, 2.0, 2.1, 2.0, 10.0, 10.1, 10.2]        # one isolated spike, then a crash-and-rebound pair
    P = pd.DataFrame({"ticker": "X", "date": days, "close": px, "volume": 1e5}); Q, s = clean(P)
    assert s["spikes"] == 1 and len(Q) == 11                                 # the 40 print is dropped
    assert s["crash_pairs"] == 1 and Q.ret.isna().sum() >= 2                  # both legs of the -80% / +400% pair are nulled
    assert Q.ret.dropna().abs().max() < 1.0

def test_clean_is_safe_with_duplicate_index_labels():
    A = synthetic_prices(3, 300, seed=1); B = synthetic_prices(3, 300, seed=2); B["ticker"] = B.ticker + "b"
    B.loc[B.index[150], "close"] = B.loc[B.index[150], "close"] * 50                   # one bad print in B only
    P = pd.concat([A, B])                                                              # duplicate index labels, as a concatenation of sources produces
    Q, s = clean(P); assert s["spikes"] == 1
    assert Q[Q.ticker.str.startswith("T") & ~Q.ticker.str.endswith("b")].ret.isna().sum() == 3        # only the first day of each A ticker is NaN

def test_panel_and_ranks():
    M, stats = build_panel(synthetic_prices()); assert set(FEATS) <= set(M.columns) and M.tradeable.any()
    R = rank_transform(M); T = R[R.tradeable]
    assert R[X_COLS].abs().max().max() <= 1.0 + 1e-9 and R[X_COLS].isna().sum().sum() == 0
    g = T[T.ym == T.ym.max()]; assert abs(g["dolvol_1m_r"].mean()) < 0.2                            # ranks are centred within the month
    assert (M.sort_values(["ticker", "ym"]).groupby("ticker").age.diff().dropna() > 0).all()           # age increases

def test_holdings_rule():
    rng = np.random.default_rng(3); n = 60; m = pd.DataFrame({"ticker": [f"T{i}" for i in range(n)], "adv": np.r_[np.full(30, 2e6), np.full(30, 2e7)], "tradeable": True, "p_RF": rng.normal(size=n)})
    h = holdings(m); assert len(h["long"]) == 12 and len(h["short"]) == 6                           # quintile of 60 longs; quintile of the 30 above $10m shorts
    assert all(m.set_index("ticker").loc[t, "adv"] >= 1e7 for t in h["short"]) and not set(h["long"]) & set(h["short"])
    assert h["forecast_gap"] > 0
    assert holdings(m.head(20)) is None                                                                # too few names -> flat
    w = weights(h, 100_000.0); assert abs(sum(v for v in w.values() if v > 0) - 100_000) < 1e-6 and abs(sum(v for v in w.values() if v < 0) + 100_000) < 1e-6

def test_sizing():
    r = pd.Series(np.random.default_rng(4).normal(0, 0.02, 500), index=pd.bdate_range("2020-01-01", periods=500)); s = sizing_vol(r)
    assert s.dropna().between(0.1, 1.0).all()
    assert abs(dollars_a_side(500_000, 0.40) - 125_000) < 1e-6 and abs(dollars_a_side(500_000, 0.05) - 400_000) < 1e-6 and abs(dollars_a_side(500_000, None) - 125_000) < 1e-6

@pytest.mark.skipif(not os.path.exists(f"{ROOT}/data/yf_prices.csv.gz") or not os.path.exists(f"{ROOT}/model/rf_2026.joblib"), reason="data not present")
def test_shipped_forecasts_reproduce():
    from adr_longshort import load_prices, load_rf, add_target, research_mask
    F = load_predictions(); ym = F.ym.max(); P = load_prices(); P = P[P.date <= (ym + 1).to_timestamp(how="end")]
    M, _ = build_panel(P[P.ticker.isin(F[F.ym == ym].ticker)]) if False else build_panel(P)
    _, rfm = load_rf(); M = add_target(M, rfm); M = rank_transform(M, research_mask(M)); m = M[(M.ym == ym) & research_mask(M)].copy()
    m["p_new"] = predict(load(f"{ROOT}/model/rf_2026.joblib"), m); x = m.merge(F[F.ym == ym][["ticker", "p_RF"]], on="ticker")
    assert len(x) == (F.ym == ym).sum() and (x.p_new - x.p_RF).abs().max() < 1e-9
