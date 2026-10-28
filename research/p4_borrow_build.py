"""Phase 4.2: per-name borrow proxy panel from free sources.

Inputs (fetched by p4_borrow_fetch_*.py into research/data/borrow/):
  ibd/<T>.json                 iBorrowDesk (IBKR) daily fee %/yr and available shares, last ~12 months only
  regsho_nasdaq_universe.csv   Nasdaq Reg SHO threshold list, every business day 2010-2026
  regsho_nyse_universe.csv     NYSE + NYSE American threshold lists, every 5th business day
  ftd_universe.csv             SEC fails-to-deliver, all semi-monthly files
  finra_si_universe.csv        FINRA consolidated short interest, bi-monthly, 2018+
Outputs: research/out/p4_borrow_panel.csv (held name-months), p4_coverage.csv, p4_fee_vs_assumed.csv, p4_fee_model.csv,
         p4_unavailable_pnl.csv; trial log research/out/trial_log_p4.csv.

Availability rule, fixed before any P&L is looked at (also written to research/out/p4_availability_rule.txt):
  a short name-month (formed at month-end ym, held in ym+1) is UNAVAILABLE if any of
   (1) iBorrowDesk observed: available shares on the last observation on/before the formation date (else the first in the
       holding month) < shares needed for the position, or the mean observed fee over the holding month > 100%/yr;
   (2) on a Reg SHO threshold list for >= 50% of the formation month's trading days;
   (3) no iBorrowDesk observation and the imputed fee > 100%/yr.
  Shares needed = (dollars_side / n_short in results/bt500k_rf_monthly.csv at the formation date) / month-end close."""
import os, sys, json, glob, numpy as np, pandas as pd
sys.path.insert(0, ".")
D = "research/data/borrow"; O = "research/out"
RULE = __doc__.split("Availability rule, fixed before any P&L is looked at (also written to research/out/p4_availability_rule.txt):")[1]
open(f"{O}/p4_availability_rule.txt", "w").write(RULE.strip() + "\n")
BASE = [(25e6, 1.0), (10e6, 4.0), (3e6, 12.0), (0, 35.0)]; HARSH = [(25e6, 2.0), (10e6, 8.0), (3e6, 25.0), (0, 60.0)]
tier = lambda a, s: next(r for lo, r in s if a >= lo)
tier_lab = lambda a: ">$25m" if a >= 25e6 else "$10-25m" if a >= 10e6 else "$3-10m" if a >= 3e6 else "<$3m"

H = pd.read_csv(f"{O}/p4_held.csv"); H["ym"] = pd.PeriodIndex(H.ym, freq="M"); H["hm"] = H.ym + 1
# ---------- characteristics from the panel (raw features; adjusted prices) ----------
M = pd.read_csv("results/panel.csv", usecols=["ticker", "ym", "price", "adv", "age", "ret_1m", "mom_12_1", "ivol_12m", "high52", "zero_1m"])
M["ym"] = pd.PeriodIndex(M.ym, freq="M"); M["ret12"] = (1 + M.mom_12_1) * (1 + M.ret_1m) - 1; M["age_m"] = np.exp(M.age)
H = H.merge(M.drop(columns="adv"), on=["ticker", "ym"], how="left")
# position size per short name at formation
BT = pd.read_csv("results/bt500k_rf_monthly.csv"); BT["ym"] = pd.PeriodIndex(pd.to_datetime(BT.date), freq="M")
BT = BT.set_index("ym"); BT = BT.reindex(pd.period_range(BT.index.min(), H.ym.max(), freq="M")).ffill()          # the last formation month carries the previous month's sizing
H["pos_usd"] = [BT.dollars_side.get(m, np.nan) / BT.n_short.get(m, np.nan) if s == "short" else BT.dollars_side.get(m, np.nan) / BT.n_long.get(m, np.nan) for m, s in zip(H.ym, H.side)]
H["shares_needed"] = H.pos_usd / H.price

# ---------- iBorrowDesk ----------
rows = []
for src, pat in (("live", f"{D}/ibd/*.json"), ("wayback", f"{D}/ibd_wayback/*.json")):
    for f in glob.glob(pat):
        t = os.path.basename(f)[:-5].split("_")[0]
        try: d = json.load(open(f))
        except Exception: continue
        for r in d.get("daily", []) or []: rows.append((t, r["date"], r.get("fee"), r.get("available"), r.get("low_available", r.get("available")), r.get("high_fee", r.get("fee")), src))
IB = pd.DataFrame(rows, columns=["ticker", "date", "fee", "avail", "low_avail", "high_fee", "src"]); IB["date"] = pd.to_datetime(IB.date)
IB = IB.sort_values("src").drop_duplicates(["ticker", "date"], keep="first"); IB["m"] = IB.date.dt.to_period("M")
print("iBorrowDesk daily rows:", IB.groupby("src").size().to_dict(), "| date range", IB.date.min().date(), "..", IB.date.max().date())
ib_tickers_ok = IB.ticker.nunique(); ib_files = len(glob.glob(f"{D}/ibd/*.json"))
ibm = IB.groupby(["ticker", "m"]).agg(fee_mean=("fee", "mean"), fee_max=("high_fee", "max"), avail_min=("low_avail", "min"), ib_days=("fee", "size")).reset_index()
# fee over the holding month
H = H.merge(ibm.rename(columns={"m": "hm"}), on=["ticker", "hm"], how="left")
# availability at formation: last obs on/before formation month-end, else first obs in the holding month
IBs = IB.sort_values("date")
def avail_at(t, ym):
    g = IBs[IBs.ticker == t]
    if g.empty: return np.nan
    end = ym.to_timestamp(how="end"); a = g[g.date <= end]
    if len(a) and (end - a.date.iloc[-1]).days <= 7: return a.avail.iloc[-1]
    b = g[(g.date > end) & (g.date <= (ym + 1).to_timestamp(how="end"))]
    return b.avail.iloc[0] if len(b) else np.nan
H["avail_form"] = [avail_at(t, m) for t, m in zip(H.ticker, H.ym)]

# ---------- Reg SHO threshold lists ----------
nd = pd.read_csv(f"{D}/regsho_nasdaq_universe.csv", parse_dates=["date"]); nd["m"] = nd.date.dt.to_period("M")
nlog = pd.read_csv(f"{D}/regsho_nasdaq_log.csv" if os.path.exists(f"{D}/regsho_nasdaq_log.csv") else f"{D}/regsho_log.csv", parse_dates=["date"]); nlog = nlog[nlog.verified.astype(bool)] if "verified" in nlog else nlog[nlog.file_ok]; nlog["m"] = nlog.date.dt.to_period("M")
ndays = nlog.groupby("m").size()
if os.path.exists(f"{D}/regsho_nyse_universe.csv"):
    ny = pd.read_csv(f"{D}/regsho_nyse_universe.csv", parse_dates=["date"]); ylog = pd.read_csv(f"{D}/regsho_nyse_log.csv", parse_dates=["date"]); ylog = ylog[ylog.ok]
else:
    print("NOTE: NYSE threshold lists not available; Nasdaq only"); ny = pd.DataFrame(columns=["date", "market", "ticker"]).astype({"date": "datetime64[ns]"}); ylog = pd.DataFrame(columns=["date", "market", "ok"]).astype({"date": "datetime64[ns]"})
ny["date"] = pd.to_datetime(ny.date); ylog["date"] = pd.to_datetime(ylog.date); ylog = ylog[ylog.ok.astype(str) == "True"]
ny["m"] = ny.date.dt.to_period("M"); ylog["m"] = ylog.date.dt.to_period("M"); ysamp = ylog.groupby(["m"]).date.nunique()
thr_n = nd.groupby(["ticker", "m"]).size() / ndays.reindex(nd.groupby(["ticker", "m"]).size().index.get_level_values(1)).values
thr_y = ny.groupby(["ticker", "m"]).date.nunique() / ysamp.reindex(ny.groupby(["ticker", "m"]).date.nunique().index.get_level_values(1)).values
thr = pd.concat([thr_n, thr_y], axis=1).max(axis=1).rename("thr_frac").reset_index().rename(columns={"m": "ym"})
H = H.merge(thr, on=["ticker", "ym"], how="left"); H["thr_frac"] = H.thr_frac.fillna(0.0)
H["regsho_covered"] = H.ym.isin(set(ndays.index)) | H.ym.isin(set(ysamp.index))
H["threshold_days"] = (H.thr_frac * 21).round(1)

# ---------- SEC fails-to-deliver ($ fails per day / $ ADV, split-invariant) ----------
F = pd.read_csv(f"{D}/ftd_universe.csv", dtype=str)
F.columns = [c.strip().upper().replace(" ", "_") for c in F.columns]; qcol = [c for c in F.columns if "QUANTITY" in c][0]
F["date"] = pd.to_datetime(F["SETTLEMENT_DATE"], format="%Y%m%d", errors="coerce"); F["q"] = pd.to_numeric(F[qcol], errors="coerce"); F["px"] = pd.to_numeric(F["PRICE"], errors="coerce")
F["usd"] = F.q * F.px; F["m"] = F.date.dt.to_period("M")
ftd_months = set(F.m.dropna())
ftd = F.groupby(["SYMBOL", "m"]).usd.sum().rename("ftd_usd").reset_index().rename(columns={"SYMBOL": "ticker", "m": "ym"})
H = H.merge(ftd, on=["ticker", "ym"], how="left"); H["ftd_covered"] = H.ym.isin(ftd_months); H.loc[H.ftd_covered, "ftd_usd"] = H.loc[H.ftd_covered, "ftd_usd"].fillna(0)
H["ftd_ratio"] = H.ftd_usd / 21 / H.adv          # mean daily $ fails over the month / $ ADV

# ---------- FINRA short interest (days to cover as reported) ----------
S = pd.read_csv(f"{D}/finra_si_universe.csv", dtype=str); S["d"] = pd.to_datetime(S.settlementDate); S["m"] = S.d.dt.to_period("M")
S["dtc"] = pd.to_numeric(S.daysToCoverQuantity, errors="coerce"); S["si"] = pd.to_numeric(S.currentShortPositionQuantity, errors="coerce")
si = S.sort_values("d").groupby(["symbolCode", "m"]).last()[["dtc", "si"]].reset_index().rename(columns={"symbolCode": "ticker", "m": "ym", "dtc": "si_dtc"})
si_months = set(S.m)
H = H.merge(si[["ticker", "ym", "si_dtc"]], on=["ticker", "ym"], how="left"); H["si_covered"] = H.ym.isin(si_months)

# ---------- assumed schedule ----------
H["fee_base"] = [tier(a, BASE) for a in H.adv]; H["fee_harsh"] = [tier(a, HARSH) for a in H.adv]; H["adv_tier"] = H.adv.map(tier_lab)
H["fee_observed"] = H.fee_mean
sh = H[H.side == "short"].copy()

# ---------- (a) coverage by source and year, held shorts ----------
sh["year"] = sh.hm.dt.year
cov = sh.groupby("year").agg(short_name_months=("ticker", "size"), ibd_fee=("fee_observed", lambda s: s.notna().sum()), regsho_covered=("regsho_covered", "sum"),
                             on_threshold_any=("thr_frac", lambda s: (s > 0).sum()), ftd_covered=("ftd_covered", "sum"), ftd_nonzero=("ftd_usd", lambda s: (s > 0).sum()),
                             si_covered=("si_dtc", lambda s: s.notna().sum()))
cov.to_csv(f"{O}/p4_coverage.csv")

# ---------- (b) observed IBKR fee vs assumed ----------
obs = H[H.fee_observed.notna()].copy(); obs["year"] = obs.hm.dt.year
def cmp(g): return pd.Series(dict(n=len(g), names=g.ticker.nunique(), ibkr_fee_median=g.fee_observed.median(), ibkr_fee_mean=g.fee_observed.mean(), ibkr_fee_p90=g.fee_observed.quantile(.9),
                                  assumed_base_mean=g.fee_base.mean(), assumed_harsh_mean=g.fee_harsh.mean(), share_fee_gt_base=(g.fee_observed > g.fee_base).mean(), share_fee_gt_harsh=(g.fee_observed > g.fee_harsh).mean(),
                                  share_fee_gt_100=(g.fee_observed > 100).mean(), share_short_avail_lt_need=(g.avail_form < g.shares_needed).mean()))
comp = pd.concat({"held shorts by ADV tier": obs[obs.side == "short"].groupby("adv_tier").apply(cmp, include_groups=False),
                  "held shorts, all": obs[obs.side == "short"].groupby(lambda _: "all").apply(cmp, include_groups=False),
                  "held longs, all": obs[obs.side == "long"].groupby(lambda _: "all").apply(cmp, include_groups=False)})
comp = pd.concat([comp, pd.concat({"held shorts by holding year": obs[obs.side == "short"].groupby("year").apply(cmp, include_groups=False)})])
comp.to_csv(f"{O}/p4_fee_vs_assumed.csv")

# ---------- fee model on the widest observed cross-section (every universe name-month with an iBorrowDesk fee) ----------
X = ibm.rename(columns={"m": "hm"}).copy(); X["ym"] = X.hm - 1
P = pd.read_csv("results/panel.csv", usecols=["ticker", "ym", "price", "adv", "age", "ret_1m", "mom_12_1", "ivol_12m"]); P["ym"] = pd.PeriodIndex(P.ym, freq="M")
P["ret12"] = (1 + P.mom_12_1) * (1 + P.ret_1m) - 1
X = X.merge(P, on=["ticker", "ym"], how="inner").merge(thr, on=["ticker", "ym"], how="left").merge(ftd, on=["ticker", "ym"], how="left").merge(si[["ticker", "ym", "si_dtc"]], on=["ticker", "ym"], how="left")
X["thr_frac"] = X.thr_frac.fillna(0); X["ftd_usd"] = X.ftd_usd.fillna(0); X["ftd_ratio"] = X.ftd_usd / 21 / X.adv
def design(d, with_si=False):
    Z = pd.DataFrame({"const": 1.0, "log_adv": np.log(d.adv.clip(lower=1e4)), "age": d.age, "ret12": d.ret12.clip(-1, 5).fillna(0), "log_price": np.log(d.price.clip(lower=0.05)),
                      "ivol": d.ivol_12m.fillna(d.ivol_12m.median() if d.ivol_12m.notna().any() else 0.05).clip(upper=0.5), "thr": d.thr_frac, "log1p_ftd": np.log1p(100 * d.ftd_ratio.fillna(0).clip(upper=5))}, index=d.index)
    if with_si: Z["log1p_dtc"] = np.log1p(d.si_dtc.fillna(0).clip(upper=50))
    return Z
X = X[X.fee_mean.notna() & (X.fee_mean > 0)].dropna(subset=["adv", "price", "age"]); y = np.log(X.fee_mean)
# inputs are winsorised to the training sample's 1st/99th percentiles before fitting and before imputing (no extrapolation outside the 2025-26 range)
BOUNDS = {w: design(X, w).quantile([0.01, 0.99]) for w in (False, True)}
def design_w(d, with_si=False): return design(d, with_si).clip(BOUNDS[with_si].iloc[0], BOUNDS[with_si].iloc[1], axis=1)
FEE_CAP = float(X.fee_mean.quantile(0.99))
fits = {}
for name, wsi in (("core", False), ("core+SI", True)):
    Z = design_w(X, wsi); b, *_ = np.linalg.lstsq(Z.values, y.values, rcond=None); yhat = Z.values @ b; r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    # grouped 5-fold CV by ticker
    tk = X.ticker.unique(); rng = np.random.default_rng(0); grp = dict(zip(tk, rng.integers(0, 5, len(tk)))); g = X.ticker.map(grp).values; pred = np.empty(len(y))
    for k in range(5):
        tr = g != k; bk, *_ = np.linalg.lstsq(Z.values[tr], y.values[tr], rcond=None); pred[~tr] = Z.values[~tr] @ bk
    cv = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(); resid_sd = np.std(y - yhat)
    sp = pd.Series(pred).corr(pd.Series(y.values), method="spearman")
    fits[name] = dict(coef=dict(zip(Z.columns, b)), r2=r2, cv_r2=cv, cv_spearman=sp, resid_sd=resid_sd, n=len(y), names=X.ticker.nunique())
pd.DataFrame({k: {**v["coef"], "R2": v["r2"], "CV_R2_by_ticker": v["cv_r2"], "CV_spearman": v["cv_spearman"], "resid_sd_log": v["resid_sd"], "n_obs": v["n"], "n_names": v["names"]} for k, v in fits.items()}).to_csv(f"{O}/p4_fee_model.csv")
# imputation with the core model (features exist 2010-26); smearing for the log retransform (mean fee)
bc = np.array(list(fits["core"]["coef"].values())); smear = np.exp(fits["core"]["resid_sd"] ** 2 / 2)
json.dump(dict(columns=list(fits["core"]["coef"].keys()), coef=list(fits["core"]["coef"].values()), lower=BOUNDS[False].iloc[0].tolist(), upper=BOUNDS[False].iloc[1].tolist(),
               smear=float(smear), fee_cap=FEE_CAP, n_obs=fits["core"]["n"], r2=fits["core"]["r2"], cv_r2=fits["core"]["cv_r2"]), open(f"{O}/p4_fee_model_frozen.json", "w"), indent=1)
H["fee_imputed"] = np.minimum(np.exp(design_w(H).values @ bc) * smear, FEE_CAP); H["fee_imputed_median"] = np.minimum(np.exp(design_w(H).values @ bc), FEE_CAP)
H["fee_best"] = H.fee_observed.fillna(H.fee_imputed); H["fee_source"] = np.where(H.fee_observed.notna(), "ibkr_observed", "imputed")

# ---------- (c) availability flag (rule above) ----------
c1 = H.fee_observed.notna() & ((H.avail_form < H.shares_needed) | (H.fee_observed > 100))
c2 = H.thr_frac >= 0.5
c3 = H.fee_observed.isna() & (H.fee_imputed > 100)
H["unavail_ibkr"] = c1; H["unavail_threshold"] = c2; H["unavail_imputed_fee"] = c3
H["available_flag"] = ~(c1 | c2 | c3); H.loc[H.side == "long", "available_flag"] = True

# ---------- P&L share in unavailable names, and net of best-estimate fee ----------
sh = H[H.side == "short"].copy(); sh["year"] = sh.hm.dt.year; sh["contrib"] = -sh.exret_next / sh.n_side            # unit short leg, gross
def pnl_tab(g):
    tot = g.contrib.sum(); u = g[~g.available_flag]
    return pd.Series(dict(name_months=len(g), unavailable=int((~g.available_flag).sum()), share_unavail=(~g.available_flag).mean(), gross_short_pnl=tot, pnl_unavail=u.contrib.sum(),
                          share_pnl_unavail=u.contrib.sum() / tot if tot else np.nan, by_ibkr=int(g.unavail_ibkr.sum()), by_threshold=int(g.unavail_threshold.sum()), by_imputed=int(g.unavail_imputed_fee.sum()),
                          fee_base_mean=g.fee_base.mean(), fee_harsh_mean=g.fee_harsh.mean(), fee_best_mean=g.fee_best.mean(), fee_best_median=g.fee_best.median()))
per = pd.concat([sh.groupby("year").apply(pnl_tab, include_groups=False), pd.DataFrame({"2010-26": pnl_tab(sh)}).T, pd.DataFrame({"2021-26": pnl_tab(sh[sh.year >= 2021])}).T])
per.to_csv(f"{O}/p4_unavailable_pnl.csv")
# unit short leg, monthly, under each fee assumption (descriptive only, no account simulation)
legm = sh.groupby("ym").apply(lambda g: pd.Series(dict(gross=(-g.exret_next).mean(), base=(-g.exret_next - g.fee_base / 1200).mean(), harsh=(-g.exret_next - g.fee_harsh / 1200).mean(),
                                                        best=(-g.exret_next - g.fee_best / 1200).mean(), avail_only_best=(-g.exret_next[g.available_flag] - g.fee_best[g.available_flag] / 1200).mean() if g.available_flag.any() else np.nan)), include_groups=False)
legm.to_csv(f"{O}/p4_short_leg_by_fee.csv")
srt = lambda x: x.mean() / x.std() * np.sqrt(12)
legsum = pd.DataFrame({c: dict(ret_pct=legm[c].mean() * 1200, sharpe=srt(legm[c].dropna()), ret_2021_26=legm.loc[legm.index >= pd.Period("2021-01", "M"), c].mean() * 1200) for c in legm.columns}).T
legsum.to_csv(f"{O}/p4_short_leg_by_fee_summary.csv")

# ---------- the window where every held short has an observed IBKR fee ----------
nsh = H[H.side == "short"].groupby("ym").size()
win = H[(H.side == "short") & H.fee_observed.notna()]; win = win[win.groupby("ym").ticker.transform("size") == win.ym.map(nsh)]
wm = win.groupby("ym").apply(lambda g: pd.Series(dict(n=len(g), gross=(-g.exret_next).mean(), net_base=(-g.exret_next - g.fee_base / 1200).mean(), net_harsh=(-g.exret_next - g.fee_harsh / 1200).mean(),
                                                     net_observed=(-g.exret_next - g.fee_observed / 1200).mean(), fee_obs_mean=g.fee_observed.mean(), fee_obs_median=g.fee_observed.median(),
                                                     n_available=int(g.available_flag.sum()),
                                                     net_observed_available_only=(-g.exret_next[g.available_flag] - g.fee_observed[g.available_flag] / 1200).mean() if g.available_flag.any() else np.nan)), include_groups=False)
wm.to_csv(f"{O}/p4_observed_window_short_leg.csv")

# ---------- (d) the panel ----------
out = H.drop(columns="hold_month").rename(columns={"hm": "hold_month"})[["ticker", "ym", "hold_month", "side", "adv", "adv_tier", "price", "age_m", "ret12", "pos_usd", "shares_needed", "fee_base", "fee_harsh",
                                              "fee_observed", "fee_imputed", "fee_imputed_median", "fee_best", "fee_source", "avail_form", "available_flag", "unavail_ibkr", "unavail_threshold", "unavail_imputed_fee",
                                              "threshold_days", "thr_frac", "ftd_ratio", "si_dtc", "regsho_covered", "ftd_covered", "si_covered", "ib_days"]]
out.sort_values(["ym", "side", "ticker"]).to_csv(f"{O}/p4_borrow_panel.csv", index=False)

# ---------- trial log ----------
log = [("p4-01", "4.2", "free-source coverage of held shorts (iBorrowDesk, Reg SHO, SEC FTD, FINRA SI)", f"ibd files {ib_files}, tickers with fee data {ib_tickers_ok}", "research/out/p4_coverage.csv"),
       ("p4-02", "4.2", "observed IBKR fee vs assumed tiered schedule, held names with iBorrowDesk data", f"short obs {int((obs.side=='short').sum())}, median fee {obs[obs.side=='short'].fee_observed.median():.2f}%", "research/out/p4_fee_vs_assumed.csv"),
       ("p4-03", "4.2", "log-fee regression on proxies (core), grouped CV by ticker", f"R2 {fits['core']['r2']:.2f}, CV R2 {fits['core']['cv_r2']:.2f}, n {fits['core']['n']}", "research/out/p4_fee_model.csv"),
       ("p4-04", "4.2", "log-fee regression on proxies + FINRA days-to-cover", f"R2 {fits['core+SI']['r2']:.2f}, CV R2 {fits['core+SI']['cv_r2']:.2f}", "research/out/p4_fee_model.csv"),
       ("p4-05", "4.2", "availability flag (pre-stated rule) and share of short P&L in unavailable names", f"2010-26 share of P&L {per.loc['2010-26','share_pnl_unavail']:.0%}, 2021-26 {per.loc['2021-26','share_pnl_unavail']:.0%}", "research/out/p4_unavailable_pnl.csv"),
       ("p4-06", "4.2", "unit short leg net of base / harsh / best-estimate fee (descriptive)", "; ".join(f"{k} {v.sharpe:.2f}" for k, v in legsum.iterrows()), "research/out/p4_short_leg_by_fee_summary.csv")]
pd.DataFrame(log, columns=["trial_id", "phase", "description", "metric_summary", "file"]).to_csv(f"{O}/trial_log_p4.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("coverage (held shorts):"); print(cov.to_string())
print("\nobserved vs assumed:"); print(comp.round(2).to_string())
print("\nfee model:"); print(pd.read_csv(f"{O}/p4_fee_model.csv", index_col=0).round(3).to_string())
print("\nunavailable P&L:"); print(per.round(3).to_string())
print("\nshort leg by fee:"); print(legsum.round(2).to_string())
print("\nobserved-fee window (every held short has an IBKR fee), unit short leg, % per month:"); print((wm * pd.Series({c: (100 if c not in ("n", "n_available", "fee_obs_mean", "fee_obs_median") else 1) for c in wm.columns})).round(2).to_string())
