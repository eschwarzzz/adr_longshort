"""Phase 4.2 extension: the borrow panel for EVERY short-eligible ticker-month (liquid & adv >= $3m, adv = each file's adv column) and every held
long, over three forecast files (union of ticker-months):
  shipped    results/predictions_rf.csv
  corrected  research/out/p0_pred_corrected.csv
  reuse      research/out/p0_pred_corrected_reuse.csv
Same method as p4_borrow_build.py: observed iBorrowDesk fee over the holding month where it exists, else the FROZEN core-model imputation
(research/out/p4_fee_model_frozen.json: coefficients, input clipping bounds, smearing factor, 99th-percentile cap; not refitted), and the same
pre-stated availability rule (research/out/p4_availability_rule.txt). Imputation inputs keep the fitted definitions: adjusted price and mean dollar
volume (adv_mean where a file has it), from results/panel.csv, else research/out/p0_panel_corrected.csv for names only in the corrected panel.
Position size: the shipped account's dollars_side/n (results/bt500k_rf_monthly.csv) for ticker-months the shipped book held, else the corrected
account's (research/out/p0_monthly_corrected.csv); shares_needed = pos_usd / price as traded (price_real where known, else adjusted price).
Output research/out/p4_borrow_panel_all.csv (same columns as p4_borrow_panel.csv plus file-membership flags), p4_all_coverage.csv,
p4_all_short_leg.csv; trial log lines appended to research/out/trial_log_p4.csv."""
import os, sys, json, glob, numpy as np, pandas as pd
sys.path.insert(0, "."); from adr_longshort import holdings
D = "research/data/borrow"; O = "research/out"
FILES = {"shipped": "results/predictions_rf.csv", "corrected": f"{O}/p0_pred_corrected.csv", "reuse": f"{O}/p0_pred_corrected_reuse.csv"}
BASE = [(25e6, 1.0), (10e6, 4.0), (3e6, 12.0), (0, 35.0)]; HARSH = [(25e6, 2.0), (10e6, 8.0), (3e6, 25.0), (0, 60.0)]
tier = lambda a, s: next(r for lo, r in s if a >= lo)
tier_lab = lambda a: ">$25m" if a >= 25e6 else "$10-25m" if a >= 10e6 else "$3-10m" if a >= 3e6 else "<$3m"
FM = json.load(open(f"{O}/p4_fee_model_frozen.json"))

# ---------- membership: short-eligible pool, held shorts, held longs, per file ----------
mem = []; held = {}
for k, f in FILES.items():
    d = pd.read_csv(f); d["ym"] = pd.PeriodIndex(d.ym, freq="M"); L = d[d.liquid]
    pool = L[L.adv >= 3e6]; mem.append(pool[["ticker", "ym"]].assign(**{f"pool_{k}": True}))
    hs, hl = [], []
    for ym, g in L.groupby("ym"):
        h = holdings(g)
        if h: hs += [(t, ym, len(h["short"])) for t in h["short"]]; hl += [(t, ym, len(h["long"])) for t in h["long"]]
    S = pd.DataFrame(hs, columns=["ticker", "ym", "n"]); Lg = pd.DataFrame(hl, columns=["ticker", "ym", "n"]); held[k] = (S, Lg, d)
    mem.append(S[["ticker", "ym"]].assign(**{f"short_{k}": True})); mem.append(Lg[["ticker", "ym"]].assign(**{f"long_{k}": True}))
A = pd.concat(mem).groupby(["ticker", "ym"]).max().reset_index()
flag_cols = [c for c in A.columns if c not in ("ticker", "ym")]; A[flag_cols] = A[flag_cols].fillna(False).astype(bool)
A["side"] = np.where(A[[c for c in flag_cols if c.startswith(("pool_", "short_"))]].any(axis=1), "short", "long")
# adv (pool definition) and adv_mean (imputation input): corrected file first, then reuse, then shipped (whose adv is the mean)
advs = []
for k in ("corrected", "reuse", "shipped"):
    d = held[k][2]; advs.append(d[["ticker", "ym", "adv"] + (["adv_mean"] if "adv_mean" in d else [])].assign(src=k))
AD = pd.concat(advs); AD["adv_mean"] = AD.adv_mean.fillna(AD.adv) if "adv_mean" in AD else AD.adv
AD = AD.drop_duplicates(["ticker", "ym"], keep="first")
A = A.merge(AD[["ticker", "ym", "adv", "adv_mean"]], on=["ticker", "ym"], how="left"); A["hm"] = A.ym + 1

# ---------- imputation inputs (fitted definitions) ----------
cols = ["ticker", "ym", "price", "age", "ret_1m", "mom_12_1", "ivol_12m"]
P1 = pd.read_csv("results/panel.csv", usecols=cols); P2 = pd.read_csv(f"{O}/p0_panel_corrected.csv", usecols=cols + ["price_real"])
for P in (P1, P2): P["ym"] = pd.PeriodIndex(P.ym, freq="M")
PF = pd.concat([P1.assign(src=1), P2.drop(columns="price_real").assign(src=2)]).drop_duplicates(["ticker", "ym"], keep="first")
A = A.merge(PF.drop(columns="src"), on=["ticker", "ym"], how="left").merge(P2[["ticker", "ym", "price_real"]], on=["ticker", "ym"], how="left")
A["ret12"] = (1 + A.mom_12_1) * (1 + A.ret_1m) - 1; A["age_m"] = np.exp(A.age)

# ---------- position size ----------
def monthly(path):
    m = pd.read_csv(path); m["ym"] = pd.PeriodIndex(pd.to_datetime(m.date), freq="M"); m = m.set_index("ym")[["dollars_side", "n_long", "n_short"]]
    return m.reindex(pd.period_range(m.index.min(), A.ym.max(), freq="M")).ffill()
MS = monthly("results/bt500k_rf_monthly.csv"); MC = monthly(f"{O}/p0_monthly_corrected.csv")
shipped_held = A.short_shipped | A.long_shipped
def size(r, M):
    if r.ym not in M.index: return np.nan
    x = M.loc[r.ym]; n = x.n_short if r.side == "short" else x.n_long
    if not n or np.isnan(n): n = x.n_short if r.side == "short" else x.n_long
    return x.dollars_side / n if n and n > 0 else np.nan
A["pos_usd"] = [size(r, MS if sh else MC) for r, sh in zip(A.itertuples(), shipped_held)]
# a shipped month with no position (flat) falls back to the corrected sizing
A.loc[A.pos_usd.isna() | (A.pos_usd == 0), "pos_usd"] = [size(r, MC) for r in A[A.pos_usd.isna() | (A.pos_usd == 0)].itertuples()]
A["px_traded"] = A.price_real.fillna(A.price); A["shares_needed"] = A.pos_usd / A.px_traded

# ---------- sources (identical treatment to p4_borrow_build.py) ----------
rows = []
for f in glob.glob(f"{D}/ibd/*.json"):
    t = os.path.basename(f)[:-5]
    try: dd = json.load(open(f))
    except Exception: continue
    for r in dd.get("daily", []) or []: rows.append((t, r["date"], r.get("fee"), r.get("available"), r.get("low_available", r.get("available")), r.get("high_fee", r.get("fee"))))
IB = pd.DataFrame(rows, columns=["ticker", "date", "fee", "avail", "low_avail", "high_fee"]); IB["date"] = pd.to_datetime(IB.date); IB = IB.drop_duplicates(["ticker", "date"]); IB["m"] = IB.date.dt.to_period("M")
ibm = IB.groupby(["ticker", "m"]).agg(fee_mean=("fee", "mean"), ib_days=("fee", "size")).reset_index().rename(columns={"m": "hm"})
A = A.merge(ibm, on=["ticker", "hm"], how="left")
IBs = IB.sort_values("date"); IBg = {t: g for t, g in IBs.groupby("ticker")}
def avail_at(t, ym):
    g = IBg.get(t)
    if g is None: return np.nan
    end = ym.to_timestamp(how="end"); a = g[g.date <= end]
    if len(a) and (end - a.date.iloc[-1]).days <= 7: return a.avail.iloc[-1]
    b = g[(g.date > end) & (g.date <= (ym + 1).to_timestamp(how="end"))]
    return b.avail.iloc[0] if len(b) else np.nan
A["avail_form"] = [avail_at(t, m) for t, m in zip(A.ticker, A.ym)]
nd = pd.read_csv(f"{D}/regsho_nasdaq_universe.csv", parse_dates=["date"]); nd["m"] = nd.date.dt.to_period("M")
nlog = pd.read_csv(f"{D}/regsho_log.csv", parse_dates=["date"]); nlog = nlog[nlog.verified.astype(bool)]; nlog["m"] = nlog.date.dt.to_period("M"); ndays = nlog.groupby("m").size()
cnt = nd.drop_duplicates(["ticker", "date"]).groupby(["ticker", "m"]).size()
thr = (cnt / ndays.reindex(cnt.index.get_level_values(1)).values).rename("thr_frac").reset_index().rename(columns={"m": "ym"})
A = A.merge(thr, on=["ticker", "ym"], how="left"); A["thr_frac"] = A.thr_frac.fillna(0.0).clip(upper=1.0); A["regsho_covered"] = A.ym.isin(set(ndays.index)); A["threshold_days"] = (A.thr_frac * 21).round(1)
F = pd.read_csv(f"{D}/ftd_universe.csv", dtype=str); F.columns = [c.strip().upper().replace(" ", "_") for c in F.columns]; qcol = [c for c in F.columns if "QUANTITY" in c][0]
F["date"] = pd.to_datetime(F["SETTLEMENT_DATE"], format="%Y%m%d", errors="coerce"); F["usd"] = pd.to_numeric(F[qcol], errors="coerce") * pd.to_numeric(F["PRICE"], errors="coerce"); F["m"] = F.date.dt.to_period("M")
ftd = F.groupby(["SYMBOL", "m"]).usd.sum().rename("ftd_usd").reset_index().rename(columns={"SYMBOL": "ticker", "m": "ym"}); ftd_months = set(F.m.dropna())
A = A.merge(ftd, on=["ticker", "ym"], how="left"); A["ftd_covered"] = A.ym.isin(ftd_months); A.loc[A.ftd_covered, "ftd_usd"] = A.loc[A.ftd_covered, "ftd_usd"].fillna(0)
A["ftd_ratio"] = A.ftd_usd / 21 / A.adv_mean
S = pd.read_csv(f"{D}/finra_si_universe.csv", dtype=str); S["d"] = pd.to_datetime(S.settlementDate); S["m"] = S.d.dt.to_period("M"); S["si_dtc"] = pd.to_numeric(S.daysToCoverQuantity, errors="coerce")
si = S.sort_values("d").groupby(["symbolCode", "m"]).last()[["si_dtc"]].reset_index().rename(columns={"symbolCode": "ticker", "m": "ym"})
A = A.merge(si, on=["ticker", "ym"], how="left"); A["si_covered"] = A.ym.isin(set(S.m))

# ---------- fees: observed, else frozen imputation ----------
A["fee_base"] = [tier(a, BASE) for a in A.adv]; A["fee_harsh"] = [tier(a, HARSH) for a in A.adv]; A["adv_tier"] = A.adv.map(tier_lab)        # panel columns on the corrected-first adv; per-book legs recompute on each file's adv
A["fee_observed"] = A.fee_mean
Z = pd.DataFrame({"const": 1.0, "log_adv": np.log(A.adv_mean.clip(lower=1e4)), "age": A.age, "ret12": A.ret12.clip(-1, 5).fillna(0), "log_price": np.log(A.price.clip(lower=0.05)),
                  "ivol": A.ivol_12m.fillna(A.ivol_12m.median()).clip(upper=0.5), "thr": A.thr_frac, "log1p_ftd": np.log1p(100 * A.ftd_ratio.fillna(0).clip(upper=5))}, index=A.index)[FM["columns"]]
Z = Z.clip(pd.Series(FM["lower"], index=FM["columns"]), pd.Series(FM["upper"], index=FM["columns"]), axis=1)
lin = Z.values @ np.array(FM["coef"])
A["fee_imputed"] = np.minimum(np.exp(lin) * FM["smear"], FM["fee_cap"]); A["fee_imputed_median"] = np.minimum(np.exp(lin), FM["fee_cap"])
A.loc[A.age.isna(), ["fee_imputed", "fee_imputed_median"]] = np.nan                       # no characteristics -> no imputation
A["fee_best"] = A.fee_observed.fillna(A.fee_imputed); A["fee_source"] = np.where(A.fee_observed.notna(), "ibkr_observed", np.where(A.fee_imputed.notna(), "imputed", "none"))
# ---------- availability: the pre-stated rule ----------
c1 = A.fee_observed.notna() & ((A.avail_form < A.shares_needed) | (A.fee_observed > 100)); c2 = A.thr_frac >= 0.5; c3 = A.fee_observed.isna() & (A.fee_imputed > 100)
A["unavail_ibkr"] = c1; A["unavail_threshold"] = c2; A["unavail_imputed_fee"] = c3; A["available_flag"] = ~(c1 | c2 | c3); A.loc[A.side == "long", "available_flag"] = True

out = A.rename(columns={"hm": "hold_month", "px_traded": "price_traded"})[["ticker", "ym", "hold_month", "side", "adv", "adv_tier", "price", "age_m", "ret12", "pos_usd", "shares_needed", "fee_base", "fee_harsh",
        "fee_observed", "fee_imputed", "fee_imputed_median", "fee_best", "fee_source", "avail_form", "available_flag", "unavail_ibkr", "unavail_threshold", "unavail_imputed_fee",
        "threshold_days", "thr_frac", "ftd_ratio", "si_dtc", "regsho_covered", "ftd_covered", "si_covered", "ib_days", "adv_mean", "price_traded"] + flag_cols]
out.sort_values(["ym", "side", "ticker"]).to_csv(f"{O}/p4_borrow_panel_all.csv", index=False)

# ---------- coverage per file, and each book's unit short leg under the fee assumptions ----------
cov = []; legs = []
for k in FILES:
    p = A[A[f"pool_{k}"]]; s = A[A[f"short_{k}"]]
    cov.append(dict(file=k, pool_name_months=len(p), pool_observed=p.fee_observed.notna().mean(), pool_imputed=(p.fee_source == "imputed").mean(), pool_none=(p.fee_source == "none").mean(),
                    pool_observed_2025_26=p[p.hm >= pd.Period("2025-11", "M")].fee_observed.notna().mean(), held_short_name_months=len(s), held_short_observed=s.fee_observed.notna().mean(),
                    held_short_unavailable=(~s.available_flag).mean(), held_short_fee_best_mean=s.fee_best.mean(), held_short_fee_best_median=s.fee_best.median(), held_short_fee_base_mean=s[['ticker', 'ym']].merge(held[k][2][['ticker', 'ym', 'adv']], on=['ticker', 'ym']).adv.map(lambda a: tier(a, BASE)).mean()))
    S_, _, d = held[k]; x = S_.merge(d[["ticker", "ym", "exret_next", "adv"]], on=["ticker", "ym"]).merge(A[["ticker", "ym", "fee_best", "available_flag"]], on=["ticker", "ym"])
    x["fee_base"] = [tier(a, BASE) for a in x.adv]; x["fee_harsh"] = [tier(a, HARSH) for a in x.adv]          # the assumed schedules on this file's own adv
    m = x.groupby("ym").apply(lambda g: pd.Series(dict(gross=(-g.exret_next).mean(), base=(-g.exret_next - g.fee_base / 1200).mean(), harsh=(-g.exret_next - g.fee_harsh / 1200).mean(),
                                                    best=(-g.exret_next - g.fee_best / 1200).mean(), avail_only_best=(-g.exret_next[g.available_flag] - g.fee_best[g.available_flag] / 1200).mean() if g.available_flag.any() else np.nan)), include_groups=False)
    pnl = (-x.exret_next / x.ym.map(x.groupby("ym").size())); share_unav = pnl[~x.available_flag.values].sum() / pnl.sum()
    for c in m.columns:
        y = m[c].dropna(); r = y[y.index >= pd.Period("2021-01", "M")]
        legs.append(dict(file=k, fee=c, ret_pct=y.mean() * 1200, sharpe=y.mean() / y.std() * np.sqrt(12), ret_2021_26=r.mean() * 1200, sharpe_2021_26=r.mean() / r.std() * np.sqrt(12), share_pnl_unavailable=share_unav))
C = pd.DataFrame(cov).set_index("file"); Lg = pd.DataFrame(legs).set_index(["file", "fee"]); C.to_csv(f"{O}/p4_all_coverage.csv"); Lg.to_csv(f"{O}/p4_all_short_leg.csv")
tl = pd.read_csv(f"{O}/trial_log_p4.csv"); tl = tl[~tl.trial_id.isin(["p4-07", "p4-08"])]
tl = pd.concat([tl, pd.DataFrame([("p4-07", "4.2", "borrow panel for every short-eligible ticker-month and held long, three forecast files, frozen fee model", f"rows {len(out)}; pool observed share " + ", ".join(f"{k} {C.loc[k,'pool_observed']:.1%}" for k in FILES), "research/out/p4_all_coverage.csv"),
                                  ("p4-08", "4.2", "unit short leg of each book under gross / base / harsh / best-estimate fee / available-only (descriptive)", "; ".join(f"{k} best {Lg.loc[(k,'best'),'sharpe']:.2f}" for k in FILES), "research/out/p4_all_short_leg.csv")],
                                columns=tl.columns)]); tl.to_csv(f"{O}/trial_log_p4.csv", index=False)
pd.set_option("display.width", 250); print(f"rows {len(out)}, tickers {out.ticker.nunique()}, no characteristics (fee_source none) {int((A.fee_source=='none').sum())}")
print(C.round(3).to_string()); print(Lg.round(2).to_string())
