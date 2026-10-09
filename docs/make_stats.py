"""Write docs/STATS.md from the files in results/ and model/, so the statistics sheet always matches the backtest outputs.

    python3 docs/make_stats.py"""
import os, json, numpy as np, pandas as pd
DOCS = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(DOCS); R = f"{ROOT}/results"
F = pd.read_csv(f"{R}/forecast_eval.csv", index_col=0); Q = pd.read_csv(f"{R}/quintile_stats_liquid_ew.csv", index_col=0); BV = pd.read_csv(f"{R}/borrow_variants_rf.csv", index_col=0)
V = pd.read_csv(f"{R}/bt500k_rf_variants.csv", index_col=0); Y = pd.read_csv(f"{R}/bt500k_rf_by_year.csv", index_col=0); SUB = pd.read_csv(f"{R}/bt500k_rf_subperiods.csv", index_col=0)
MC = json.load(open(f"{R}/bt500k_rf_mc.json")); V5 = pd.read_csv(f"{R}/bt500k_rf_5y_variants.csv", index_col=0); Y5 = pd.read_csv(f"{R}/bt500k_rf_5y_by_year.csv", index_col=0); MC5 = json.load(open(f"{R}/bt500k_rf_5y_mc.json"))
QR = pd.read_csv(f"{R}/quintile_returns_rf.csv"); M = pd.read_csv(f"{R}/bt500k_rf_monthly.csv", index_col=0, parse_dates=True); D = pd.read_csv(f"{R}/bt500k_rf_daily.csv", index_col=0, parse_dates=True)
H = "Headline: $10m short leg, 10% vol target"; E = "Ensemble forecasts, 10% vol"; has_ens = E in V.index
r = Q.loc["RF"]; e = Q.loc["ENS"] if "ENS" in Q.index else None
q = QR.copy(); q["ls"] = q.q5 - q.q1; q["year"] = pd.PeriodIndex(q.ym, freq="M").year; ls_yr = q.groupby("year").ls.apply(lambda x: ((1 + x).prod() - 1) * 100)
yr_txt = ", ".join(f"{y}{' (to August)' if y == 2026 else ''} {v:+.0f}" for y, v in ls_yr.items())
def row(v, k): return V.loc[k, v]
def pct(x, d=1): return f"{x:.{d}f}%"
def usd(x): return ("−" if x < 0 else "") + (f"${abs(x)/1e6:.2f}m" if abs(x) >= 1e6 else f"${abs(x)/1e3:,.0f}k")
def sgn(x, d=1): return f"{x:+.{d}f}"
hb = MC["bootstrap_sharpe"]["Headline"]; hp = MC["bootstrap_pnl"]["Headline"]; nl = MC["nulls"]; ds = MC["deflated"]; st = MC.get("stale", {})
hb5 = MC5["bootstrap_sharpe"]["Headline"]; hp5 = MC5["bootstrap_pnl"]["Headline"]; nl5 = MC5["nulls"]; ds5 = MC5["deflated"]
eq = D.equity; dd = eq / eq.cummax() - 1; trough = dd.idxmin(); peak = eq.loc[:trough].idxmax(); rec = eq.loc[trough:][eq.loc[trough:] >= eq.loc[peak]].index
mm = M[M.n_short > 0]; best5 = M.pnl.nlargest(5).sum() / M.pnl.sum(); best10 = M.pnl.nlargest(10).sum() / M.pnl.sum()
yrs = Y.dropna(); yr_acct = ", ".join(f"{int(y)}{' (to August)' if int(y) == 2026 else ''} {v:+.1f}" for y, v in yrs.ret_pct.items())
flat = M.index[M.exposure == 0]; flat_txt = (f"{len(flat)} (" + ", ".join(pd.Series(flat).dt.strftime("%b %Y")) + ")") if len(flat) else "0"
from scipy.stats import norm
T = pd.read_csv(f"{R}/research_quintile_returns.csv"); trial = T.assign(ls=T.q5 - T.q1).groupby("strategy").ls.apply(lambda s: s.mean() / s.std()); NT = len(trial); dsr_k = f"dsr_{NT}"; _g = 0.5772156649
def expmax(nt): return np.sqrt(trial.var()) * ((1 - _g) * norm.ppf(1 - 1 / nt) + _g * norm.ppf(1 - 1 / (nt * np.e))) * np.sqrt(12)
# the five-year account
h5 = V5.loc[H]; y5 = Y5.dropna(); yr5_txt = ", ".join(f"{int(y)} {('(Sep–Dec) ' if int(y) == 2021 else '(to August) ' if int(y) == 2026 else '')}{v:+.1f}%" for y, v in y5.ret_pct.items())
D5 = pd.read_csv(f"{R}/bt500k_rf_5y_daily.csv", index_col=0, parse_dates=True); M5 = pd.read_csv(f"{R}/bt500k_rf_5y_monthly.csv", index_col=0, parse_dates=True)
eq5 = D5.equity; dd5 = eq5 / eq5.cummax() - 1; tr5 = dd5.idxmin(); pk5 = eq5.loc[:tr5].idxmax(); rec5 = eq5.loc[tr5:][eq5.loc[tr5:] >= eq5.loc[pk5]].index
roll = M.ret.rolling(60).apply(lambda x: x.mean() / x.std() * np.sqrt(12)); roll_txt = f"{roll.iloc[-1]:.2f} today against a median of {roll.median():.2f} and a range of {roll.min():.2f} to {roll.max():.2f} over {roll.notna().sum()} windows"
lo5 = V5.loc["Long leg only, 10% vol"]
ens_col = (lambda k: f" {usd(row('final_equity', E)) if k == 'fe' else ''}")
def vline(k, extra=""):
    v = V.loc[k]; return f"| {extra or k} | {usd(v.final_equity)} | {pct(v.cagr_pct)} / {pct(v.vol_pct)} | {v.sharpe:.2f} / {v.sharpe_monthly:.2f} | −{-v.maxdd_pct:.1f}% | {v.worst_month_pct:.1f}% / {v.hit_pct:.0f}% | {v.avg_side_pct:.0f}% |"
out = f"""# Backtest statistics: random-forest Chinese ADR long-short

All numbers are out of sample in the sense that no model saw data from its test year; the design (features, universe, model menu, the $10m floor) was fixed with the research and its literature in view. Window January 2010 to August 2026. Generated from the files in `results/` by `docs/make_stats.py`.

## 1. Forecast quality

All tradeable names, 2010–26. File: `forecast_eval.csv`.

| | R² vs zero forecast | Rank IC (t) | Rank IC, liquid names (t) |
|---|---|---|---|
| Random forest | {F.loc['RF', 'r2_oos_pct']:.2f}% | {F.loc['RF', 'rank_ic']:.3f} ({F.loc['RF', 'ic_t']:.1f}) | {F.loc['RF', 'rank_ic_liquid']:.3f} ({F.loc['RF', 'ic_t_liquid']:.1f}) |
""" + (f"| Ensemble of five models, for reference | {F.loc['ENS', 'r2_oos_pct']:.2f}% | {F.loc['ENS', 'rank_ic']:.3f} ({F.loc['ENS', 'ic_t']:.1f}) | {F.loc['ENS', 'rank_ic_liquid']:.3f} ({F.loc['ENS', 'ic_t_liquid']:.1f}) |\n" if "ENS" in F.index else "") + (f"| Huber OLS, for reference | {F.loc['OLS-H', 'r2_oos_pct']:.2f}% | {F.loc['OLS-H', 'rank_ic']:.3f} ({F.loc['OLS-H', 'ic_t']:.1f}) | {F.loc['OLS-H', 'rank_ic_liquid']:.3f} ({F.loc['OLS-H', 'ic_t_liquid']:.1f}) |\n" if "OLS-H" in F.index else "") + f"""
## 2. Quintile portfolios, unrestricted short leg

Liquid universe (≥ $1m a day), equal weight, {int(r.months)} months, {r.avg_names:.0f} names a month.

Files: `quintile_stats_liquid_ew.csv`; monthly series in `quintile_returns_rf.csv`.

| | Random forest |
|---|---|
| Long-short return, annualised | {sgn(r.ls_ret_pct)}% |
| Long-short Sharpe, gross / net of 20 bp a side and 3% borrow | {r.ls_sharpe:.2f} / {r.ls_net_sharpe:.2f} (t {r.ls_t:.1f}) |
| Long-short worst drawdown | −{-r.ls_maxdd_pct:.0f}% |
| Top quintile / bottom quintile excess return | {sgn(r.q5_ret_pct)}% / {sgn(r.q1_ret_pct)}% a year |
| Equal-weight universe excess return | {sgn(r.ew_ret_pct)}% a year |
| Top quintile minus equal-weight universe, Sharpe | {r.long_excess_sharpe:.2f} |
| Turnover, long / short leg | {r.turnover_q5*100:.0f}% / {r.turnover_q1*100:.0f}% a month |
| Sub-period Sharpe 2010–15 / 2016–20 / 2021–26 | {r['ls_sharpe_2010-15']:.2f} / {r['ls_sharpe_2016-20']:.2f} / {r['ls_sharpe_2021-26']:.2f} |
| Calendar years, long-short gross (%) | {yr_txt} |

## 3. Borrow at retail scale and the $10m short-leg rule

File: `borrow_variants_rf.csv`.

Schedules by daily dollar volume, assumed not quoted: base 1% above $25m, 4% at $10–25m, 12% at $3–10m, 35% below $3m; harsh 2 / 8 / 25 / 60%.

| Short-leg rule | Gross Sharpe | Average borrow | Net Sharpe (base / harsh) | Net 2021–26 | Shorts a month |
|---|---|---|---|---|---|
| All liquid names, flat 3% borrow | {BV.loc['flat borrow 3%', 'sharpe']:.2f} | 3% | {BV.loc['flat borrow 3%', 'net_sharpe']:.2f} | {BV.loc['flat borrow 3%', 'net_2021_26']:.2f} | {BV.loc['flat borrow 3%', 'n_short']:.0f} |
| All liquid names, tiered | {BV.loc['tiered borrow (base)', 'sharpe']:.2f} | {BV.loc['tiered borrow (base)', 'borrow_pct']:.0f} / {BV.loc['tiered borrow (harsh)', 'borrow_pct']:.0f}% | {BV.loc['tiered borrow (base)', 'net_sharpe']:.2f} / {BV.loc['tiered borrow (harsh)', 'net_sharpe']:.2f} | {BV.loc['tiered borrow (base)', 'net_2021_26']:.2f} / {BV.loc['tiered borrow (harsh)', 'net_2021_26']:.2f} | {BV.loc['tiered borrow (base)', 'n_short']:.0f} |
""" + "".join(f"| {'**' if lab == '$10m' else ''}Names above {lab} a day{' (exported rule)**' if lab == '$10m' else ''} | {'**' if lab == '$10m' else ''}{BV.loc[f'short leg >= {lab}/day, tiered (base)', 'sharpe']:.2f}{'**' if lab == '$10m' else ''} | {'**' if lab == '$10m' else ''}{BV.loc[f'short leg >= {lab}/day, tiered (base)', 'borrow_pct']:.0f} / {BV.loc[f'short leg >= {lab}/day, tiered (harsh)', 'borrow_pct']:.0f}%{'**' if lab == '$10m' else ''} | {'**' if lab == '$10m' else ''}{BV.loc[f'short leg >= {lab}/day, tiered (base)', 'net_sharpe']:.2f} / {BV.loc[f'short leg >= {lab}/day, tiered (harsh)', 'net_sharpe']:.2f}{'**' if lab == '$10m' else ''} | {'**' if lab == '$10m' else ''}{BV.loc[f'short leg >= {lab}/day, tiered (base)', 'net_2021_26']:.2f} / {BV.loc[f'short leg >= {lab}/day, tiered (harsh)', 'net_2021_26']:.2f}{'**' if lab == '$10m' else ''} | {'**' if lab == '$10m' else ''}{BV.loc[f'short leg >= {lab}/day, tiered (base)', 'n_short']:.0f}{'**' if lab == '$10m' else ''} |\n" for lab in ["$3m", "$10m", "$25m"]) + f"""| Long leg, short PGJ at 1% | {BV.loc['long leg, short PGJ', 'sharpe']:.2f} | 1% | {BV.loc['long leg, short PGJ', 'net_sharpe']:.2f} | {BV.loc['long leg, short PGJ', 'net_2021_26']:.2f} | |

## 4. The $500,000 account, 2010–26

Files: `bt500k_rf_variants.csv`, `bt500k_rf_by_year.csv`, `bt500k_rf_monthly.csv`; figures 2 to 4.

Daily P&L on the monthly holdings; dollars a side = min(80%, 10% ÷ σ) × equity with σ from the unit long-short's 60-day volatility blended with its long-run mean; 20 bp a side; borrow per name by volume; free cash at the bill less 0.5%, short proceeds above $100k at the bill less 1.25%; Reg T margin.

| | Random forest (headline) |{" Same account on the ensemble forecasts |" if has_ens else ""}
|---|---|{"---|" if has_ens else ""}
| Final equity from $500,000 | **{usd(row('final_equity', H))}** |{f" {usd(row('final_equity', E))} |" if has_ens else ""}
| CAGR / volatility | **{pct(row('cagr_pct', H))} / {pct(row('vol_pct', H))}** |{f" {pct(row('cagr_pct', E))} / {pct(row('vol_pct', E))} |" if has_ens else ""}
| Sharpe, daily / monthly | **{row('sharpe', H):.2f} / {row('sharpe_monthly', H):.2f}** |{f" {row('sharpe', E):.2f} / {row('sharpe_monthly', E):.2f} |" if has_ens else ""}
| Worst drawdown | −{-row('maxdd_pct', H):.1f}% ({peak:%b %Y} to {trough:%b %Y}, recovered {rec[0]:%b %Y}) |{f" −{-row('maxdd_pct', E):.1f}% |" if has_ens else ""}
| Worst month / hit rate | {row('worst_month_pct', H):.1f}% / {row('hit_pct', H):.0f}% |{f" {row('worst_month_pct', E):.1f}% / {row('hit_pct', E):.0f}% |" if has_ens else ""}
| Sub-period Sharpe 2010–15 / 2016–20 / 2021–26 | {SUB.loc[H, '2010-15']:.2f} / {SUB.loc[H, '2016-20']:.2f} / {SUB.loc[H, '2021-26']:.2f} |{f" {SUB.loc[E, '2010-15']:.2f} / {SUB.loc[E, '2016-20']:.2f} / {SUB.loc[E, '2021-26']:.2f} |" if has_ens else ""}
| Exposure a side, average (range) | {row('avg_side_pct', H):.0f}% ({mm.exposure.min()*100:.0f} to {row('max_side_pct', H):.0f}%) |{f" {row('avg_side_pct', E):.0f}% |" if has_ens else ""}
| Margin pledged, average / peak (Reg T initial); maintenance peak; call days | {row('avg_margin_pct', H):.0f}% / {row('peak_margin_pct', H):.0f}%; {row('peak_maint_pct', H):.0f}%; {int(row('call_days', H))} |{" |" if has_ens else ""}
| Commissions and spread / borrow ({row('borrow_rate_pct', H):.1f}% a year) / cash interest earned | {usd(row('fees_usd', H))} / {usd(row('borrow_usd', H))} / +{usd(row('interest_usd', H))} |{" |" if has_ens else ""}
| Flat months (too few names) | {flat_txt} |{" |" if has_ens else ""}

Calendar years, return on the account (%): {yr_acct}. The best five months are {best5:.0%} of the P&L, the best ten {best10:.0%}.

Sizing and cost variants (final equity, CAGR, Sharpe daily, worst drawdown): 15% vol target {usd(row('final_equity', 'Headline, 15% vol target'))}, {pct(row('cagr_pct', 'Headline, 15% vol target'))}, {row('sharpe', 'Headline, 15% vol target'):.2f}, −{-row('maxdd_pct', 'Headline, 15% vol target'):.0f}%; 20% target {usd(row('final_equity', 'Headline, 20% vol target'))}, {pct(row('cagr_pct', 'Headline, 20% vol target'))}, {row('sharpe', 'Headline, 20% vol target'):.2f}, −{-row('maxdd_pct', 'Headline, 20% vol target'):.0f}%; fixed 50% a side {usd(row('final_equity', 'Headline, fixed 50% a side'))}, {pct(row('cagr_pct', 'Headline, fixed 50% a side'))}, {row('sharpe', 'Headline, fixed 50% a side'):.2f}, −{-row('maxdd_pct', 'Headline, fixed 50% a side'):.0f}%; fixed 80% a side {usd(row('final_equity', 'Headline, fixed 80% a side'))}, {pct(row('cagr_pct', 'Headline, fixed 80% a side'))}, {row('sharpe', 'Headline, fixed 80% a side'):.2f}, −{-row('maxdd_pct', 'Headline, fixed 80% a side'):.0f}% with {int(row('call_days', 'Headline, fixed 80% a side'))} margin-call days; harsh borrow {row('sharpe', 'Headline, harsh borrow'):.2f}; flat 8% borrow {row('sharpe', 'Headline, flat 8% borrow'):.2f}; no borrow {row('sharpe', 'Headline, no borrow cost'):.2f}; 10 bp costs {row('sharpe', 'Headline, cost 10 bp/side'):.2f}; 40 bp costs {row('sharpe', 'Headline, cost 40 bp/side'):.2f}.

## 5. Monte Carlo and nulls

File: `bt500k_rf_mc.json`; figure 4.

Block bootstrap, 12-month blocks, 10,000 paths of {len(M)} months: Sharpe median {hb['p50']:.2f}, 90% interval {hb['p05']:.2f} to {hb['p95']:.2f}, above zero in {'every path' if hb['prob_gt0'] >= 0.9995 else f'{hb["prob_gt0"]:.0%} of paths'}; final equity {usd(hp['final_p05'])} (5th percentile), {usd(hp['final_p50'])} (median), {usd(hp['final_p95'])} (95th); CAGR {hp['cagr_p05']:.1f} to {hp['cagr_p95']:.1f}%; worst drawdown median −{-hp['maxdd_p50']:.0f}%, one path in twenty worse than −{-hp['maxdd_p05']:.0f}%, one in a hundred worse than −{-hp['maxdd_p01']:.0f}%; one-year P&L {usd(hp['pnl1y_p05'])} / {usd(hp['pnl1y_p50'])} / {usd(hp['pnl1y_p95'])} at the 5th / 50th / 95th percentiles, {hp['prob_loss_1y']:.0%} chance of a losing year; {hp['prob_loss_5y']:.1%} chance of being down after five years. Beats the long leg alone in {hb['prob_beats_longonly']:.0%} of paired paths and the full-short-leg account in {hb['prob_beats_full']:.0%}. The bootstrap resamples this one history; it measures sampling noise around these sixteen years, not the chance that the next cycle differs.

Random-sort null: 500 random rankings through the same universe, construction, exposure path, costs and borrow give Sharpe {nl['null_mean']:.2f} on average (sd {nl['null_sd']:.2f}), 95th percentile {nl['null_p95']:.2f}, maximum {nl['null_max']:+.2f}; {"none reaches" if nl['p_null'] == 0 else f"{nl['p_null']:.1%} reach"} the observed {nl['observed']:.2f}. The null sits below zero because a random version of the rule is long an average liquid name and short an average name above $10m a day, a drag the forecasts have to overcome first.

Stale forecasts (rank by the forecast made s months earlier; unit long-short, gross Sharpe): {", ".join(f"{v['gross_sharpe']:.2f} at s = {k}" if k == '0' else f"{v['gross_sharpe']:.2f} at {k}" for k, v in st.items() if k in ('0', '1', '2', '3', '6', '12'))}.

Deflated Sharpe: with the dispersion of the {NT} strategies examined on the research panel, the expected maximum Sharpe under a no-skill null is {expmax(NT):.2f} for {NT} trials and {expmax(70):.2f} for 70. The account's monthly Sharpe of {ds['Headline']['sharpe']:.2f} (skew {ds['Headline']['skew']:.2f}, kurtosis {ds['Headline']['kurt']:.1f}) gives a deflated-Sharpe probability of {ds['Headline'][dsr_k]:.2f} at {NT} trials and {ds['Headline']['dsr_70']:.2f} at 70. The random forest was the best of the eight models on this panel, so the 70-trial reading is the honest one.

## 6. The last five years as a fresh account

September 2021 to August 2026. Files: `bt500k_rf_5y_*`; figures 5 and 6, `bt500k_rf_5y_pages.pdf`.

Final equity {usd(h5.final_equity)} (+{usd(h5.pnl_usd)}, of which {usd(h5.interest_usd)} is cash interest), CAGR {pct(h5.cagr_pct)}, volatility {pct(h5.vol_pct)}, Sharpe {h5.sharpe:.2f} daily / {h5.sharpe_monthly:.2f} monthly, worst drawdown −{-h5.maxdd_pct:.1f}% ({pk5:%B %Y} to {tr5:%B %Y}, {"recovered " + rec5[0].strftime("%B %Y") if len(rec5) else "not yet recovered"}), worst month {h5.worst_month_pct:.1f}%, hit rate {h5.hit_pct:.0f}%; exposure {h5.avg_side_pct:.0f}% a side on average; borrow {h5.borrow_rate_pct:.1f}%; by year {yr5_txt}. Bootstrap Sharpe median {hb5['p50']:.2f}, 90% interval {hb5['p05']:.2f} to {hb5['p95']:.2f}; final equity {usd(hp5['final_p05'])} to {usd(hp5['final_p95'])}; worst drawdown median −{-hp5['maxdd_p50']:.0f}%, one in twenty −{-hp5['maxdd_p05']:.0f}%; {hp5['prob_loss_5y']:.1%} chance of being down after five years. Random-sort null on the 60 months: {"none of 300 random sorts reaches" if nl5['p_null'] == 0 else f"{nl5['p_null']:.1%} of 300 random sorts reach"} {nl5['observed']:.2f} (maximum {nl5['null_max']:.2f}). Deflated Sharpe {ds5['Headline'][dsr_k]:.2f} / {ds5['Headline']['dsr_70']:.2f}. The rolling five-year Sharpe of the 2010-start account is {roll_txt}. The long leg alone {"lost money" if lo5.pnl_usd < 0 else "made " + usd(lo5.pnl_usd)} over the five years ({lo5.cagr_pct:+.1f}% a year): the result is the short leg.

## 7. Cross-checks and limits
- Price data: Yahoo and an independent archive agree on 99.95% of overlapping stock-months. Two data bugs were found and fixed in review (an asymmetric return filter that turned bad prints into −100% months, and a label-indexing error in the cleaning step that nulled unrelated days when sources were concatenated); the numbers here are from the corrected pipeline.
- Survivorship: names that delisted after April 2020, other than twelve SOEs carried through Hong Kong, are missing for their final months. The volume-weighted universe correlates 0.92 with PGJ and trails it by 4% a year in 2021–26, so the survivor premium is not driving the recent numbers; the small-cap delistings sit in the short leg and remain unmeasured.
- Concentration: the pool above $10m a day was 10 to 25 names until 2017, so the short leg was its minimum of five names; it has been 7 to 12 since 2018.
"""
import re; out = re.sub(r"(?<![\w\u2013-])-(?=\d)", "\u2212", out)
open(f"{DOCS}/STATS.md", "w").write(out); print("docs/STATS.md written")
