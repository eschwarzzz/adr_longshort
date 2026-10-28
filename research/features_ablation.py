"""Feature impact study for the random forest: drop-one, drop/keep family, greedy forward selection (chosen on validation only), and
out-of-sample permutation importance. Hyperparameters per year are fixed to the shipped ones (model/rf_hyperparams_by_year.csv).

    python research/features_ablation.py [ablate] [greedy] [perm]      # default: all three

Outputs research/out/ablation_*.csv."""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "."); from adr_longshort import rank_transform, holdings, FEATS, X_COLS
from sklearn.ensemble import RandomForestRegressor
O = "research/out"; STAGES = set(sys.argv[1:]) or {"ablate", "greedy", "perm"}; t0 = time.time()
M = pd.read_csv("results/panel.csv"); M["ym"] = pd.PeriodIndex(M.ym, freq="M"); M = M[M.in_sample].copy(); M = rank_transform(M); M["liquid"] = M.adv >= 1e6
M = M.reset_index(drop=True); HP = pd.read_csv("model/rf_hyperparams_by_year.csv", index_col=0); YEARS = list(range(2010, 2027))
FAM = {"trend": ["ret_1m", "mom_6_1", "mom_12_1", "mom_36_13"], "risk": ["vol_1m", "vol_3m", "ivol_12m", "beta_12m", "maxret_1m", "minret_1m"],
       "liquidity": ["dolvol_1m", "dolvol_trend", "illiq_1m", "zero_1m"], "other": ["high52", "age"]}
BASE = [(25e6, .01), (10e6, .04), (3e6, .12), (0, .35)]; COST = 0.002
tier = lambda a: next(r for lo, r in BASE if a >= lo)
def split(Y): return M[(M.year >= 2001) & (M.year <= Y - 3)], M[(M.year >= Y - 2) & (M.year <= Y - 1)], M[M.year == Y]
def wins(y, ref): lo, hi = ref.quantile([.01, .99]); return y.clip(lo, hi)
def rf(Y, n=300): return RandomForestRegressor(n_estimators=n, max_depth=int(HP.loc[Y, "max_depth"]), min_samples_leaf=int(HP.loc[Y, "min_samples_leaf"]), max_features=0.5, n_jobs=-1, random_state=0)
def fit_predict(feats, n=300):
    """Annual refits with the given feature subset; returns the test-year forecast for every in-sample row 2010-26 and the per-year models."""
    cols = [f + "_r" for f in feats]; p = pd.Series(np.nan, index=M.index); models = {}
    for Y in YEARS:
        tr, va, te = split(Y); m = rf(Y, n).fit(tr[cols].values, wins(tr.exret_next, tr.exret_next).values); p.loc[te.index] = m.predict(te[cols].values); models[Y] = m
    return p, models
def monthly_ic(df, c): return df.groupby("ym")[[c, "exret_next"]].apply(lambda g: g[c].rank().corr(g.exret_next.rank()))
def ls_series(p):
    """Monthly gross and net (20 bp a side, base tiered borrow) unit long-short with the $3m short floor."""
    d = M.assign(p=p); rows = []; p5 = p1 = set()
    for ym, g in d[d.liquid & d.p.notna()].groupby("ym"):
        h = holdings(g, "p")
        if h is None: continue
        lo = g[g.ticker.isin(h["long"])]; sh = g[g.ticker.isin(h["short"])]; s5, s1 = set(h["long"]), set(h["short"])
        to = (1 - len(s5 & p5) / len(s5) if p5 else 1) + (1 - len(s1 & p1) / len(s1) if p1 else 1); p5, p1 = s5, s1
        gr = lo.exret_next.mean() - sh.exret_next.mean(); rows.append((ym, gr, gr - COST * 2 * to - np.mean([tier(a) for a in sh.adv]) / 12))
    return pd.DataFrame(rows, columns=["ym", "gross", "net"]).set_index("ym")
sr = lambda x: x.mean() / x.std() * np.sqrt(12)
def evaluate(p, name, base_ls=None):
    d = M.assign(p=p).dropna(subset=["p"]); ic = monthly_ic(d, "p"); icl = monthly_ic(d[d.liquid], "p"); ls = ls_series(p); g = ls.gross
    r = dict(variant=name, ic=ic.mean(), ic_t=ic.mean() / ic.std() * np.sqrt(len(ic)), ic_liq=icl.mean(), r2_pct=(1 - ((d.exret_next - d.p) ** 2).sum() / (d.exret_next ** 2).sum()) * 100,
             ls_ret=g.mean() * 1200, ls_sharpe=sr(g), net_sharpe=sr(ls.net), sh_10_15=sr(g.loc["2010":"2015"]), sh_16_20=sr(g.loc["2016":"2020"]), sh_21_26=sr(g.loc["2021":]))
    if base_ls is not None:
        x = (g - base_ls.gross).dropna(); r["diff_t_vs_base"] = x.mean() / x.std() * np.sqrt(len(x))
    return r, ls

# baseline: must reproduce the shipped forecasts
shipped = pd.read_csv("results/predictions_rf.csv"); shipped["ym"] = pd.PeriodIndex(shipped.ym, freq="M")
pb, MODELS = fit_predict(FEATS); chk = M.assign(p=pb)[["ticker", "ym", "p"]].merge(shipped[["ticker", "ym", "p_RF"]], on=["ticker", "ym"])
print(f"baseline vs shipped p_RF: rows {len(chk)}, max |diff| {(chk.p - chk.p_RF).abs().max():.1e} [{time.time()-t0:.0f}s]", flush=True)
rb, LSB = evaluate(pb, "baseline (16)"); print({k: round(v, 3) if isinstance(v, float) else v for k, v in rb.items()}, flush=True)
pd.set_option("display.width", 250)

if "ablate" in STAGES:
    variants = {f"drop {f}": [x for x in FEATS if x != f] for f in FEATS}
    variants.update({f"drop family {k}": [x for x in FEATS if x not in v] for k, v in FAM.items()})
    variants.update({f"only family {k}": v for k, v in FAM.items()})
    variants.update({"only zero_1m+high52": ["zero_1m", "high52"], "only top4 (zero,high52,ivol,mom_6_1)": ["zero_1m", "high52", "ivol_12m", "mom_6_1"]})
    out = [dict(rb, diff_t_vs_base=0.0)]
    for name, fs in variants.items():
        p, _ = fit_predict(fs); r, _ = evaluate(p, name, LSB); out.append(r); print(f"{name}: IC {r['ic']:.3f}, LS {r['ls_sharpe']:.2f}, t {r['diff_t_vs_base']:+.2f} [{time.time()-t0:.0f}s]", flush=True)
    A = pd.DataFrame(out).set_index("variant"); A.to_csv(f"{O}/ablation_variants.csv"); print(A.round(3).to_string(), flush=True)

if "perm" in STAGES:
    rng = np.random.default_rng(0); out = []; NS = 3
    for f in FEATS:
        c = f + "_r"; ics = []; lss = []
        for s in range(NS):
            p = pd.Series(np.nan, index=M.index)
            for Y in YEARS:
                te = M[M.year == Y].copy(); te[c] = te.groupby("ym")[c].transform(lambda x: rng.permutation(x.values)); p.loc[te.index] = MODELS[Y].predict(te[X_COLS].values)
            d = M.assign(p=p); ics.append(monthly_ic(d, "p").mean()); lss.append(sr(ls_series(p).gross))
        out.append(dict(feature=f, ic_drop=rb["ic"] - np.mean(ics), ls_sharpe_drop=rb["ls_sharpe"] - np.mean(lss), ls_sharpe_sd=np.std(lss)))
        print(f"perm {f}: IC drop {out[-1]['ic_drop']:.4f}, LS drop {out[-1]['ls_sharpe_drop']:.2f} [{time.time()-t0:.0f}s]", flush=True)
    imp = pd.DataFrame([dict(feature=f, **{f"imp_{Y}": MODELS[Y].feature_importances_[i] for Y in YEARS}) for i, f in enumerate(FEATS)]).set_index("feature")
    PI = pd.DataFrame(out).set_index("feature").join(imp.mean(axis=1).rename("mdi_mean_2010_26")).sort_values("ls_sharpe_drop", ascending=False)
    PI.to_csv(f"{O}/ablation_permutation.csv"); imp.to_csv(f"{O}/ablation_mdi_by_year.csv"); print(PI.round(3).to_string(), flush=True)

if "greedy" in STAGES:
    # per test year: forward selection on that year's validation window (mean monthly rank IC), 100 trees for the search; no test data used
    KMAX = 8; chosen = {}
    for Y in YEARS:
        tr, va, _ = split(Y); y = wins(tr.exret_next, tr.exret_next).values; sel = []
        for k in range(KMAX):
            best = None
            for f in [x for x in FEATS if x not in sel]:
                cols = [g + "_r" for g in sel + [f]]; m = rf(Y, 100).fit(tr[cols].values, y); v = va.assign(p=m.predict(va[cols].values)); s = monthly_ic(v, "p").mean()
                if best is None or s > best[1]: best = (f, s)
            sel.append(best[0])
        chosen[Y] = sel; print(f"greedy {Y}: {sel} [{time.time()-t0:.0f}s]", flush=True)
    pd.DataFrame(chosen).T.rename(columns=lambda k: f"step{k+1}").to_csv(f"{O}/ablation_greedy_order.csv")
    out = []
    for k in range(1, KMAX + 1):
        p = pd.Series(np.nan, index=M.index)
        for Y in YEARS:
            tr, va, te = split(Y); cols = [g + "_r" for g in chosen[Y][:k]]; p.loc[te.index] = rf(Y).fit(tr[cols].values, wins(tr.exret_next, tr.exret_next).values).predict(te[cols].values)
        r, _ = evaluate(p, f"greedy k={k}", LSB); out.append(r)
    G = pd.DataFrame(out).set_index("variant"); G.to_csv(f"{O}/ablation_greedy_path.csv"); print(G.round(3).to_string())
    first = pd.Series([f for Y in YEARS for f in chosen[Y][:4]]).value_counts(); print("features in the first four picks, count of years (of 17):", first.to_dict())
print(f"DONE [{time.time()-t0:.0f}s]")
