# Backtest statistics: random-forest Chinese ADR long-short

All numbers are out of sample in the sense that no model saw data from its test year; the design (features, universe, model menu, the $3m floor) was fixed with the research and its literature in view. Window January 2010 to August 2026. Generated from the files in `results/` by `docs/make_stats.py`.

## 1. Forecast quality

All tradeable names, 2010–26. File: `forecast_eval.csv`.

| | R² vs zero forecast | Rank IC (t) | Rank IC, liquid names (t) |
|---|---|---|---|
| Random forest | 0.35% | 0.080 (7.0) | 0.081 (6.1) |
| Ensemble of five models, for reference | 0.31% | 0.087 (8.2) | 0.089 (7.0) |
| Huber OLS, for reference | 0.44% | 0.115 (9.7) | 0.107 (7.3) |

## 2. Quintile portfolios, unrestricted short leg

Liquid universe (≥ $1m a day), equal weight, 191 months, 62 names a month.

Files: `quintile_stats_liquid_ew.csv`; monthly series in `quintile_returns_rf.csv`.

| | Random forest |
|---|---|
| Long-short return, annualised | +41.1% |
| Long-short Sharpe, gross / net of 20 bp a side and 3% borrow | 1.16 / 0.93 (t 4.6) |
| Long-short worst drawdown | −37% |
| Top quintile / bottom quintile excess return | +13.9% / −27.2% a year |
| Equal-weight universe excess return | −1.5% a year |
| Top quintile minus equal-weight universe, Sharpe | 0.89 |
| Turnover, long / short leg | 48% / 58% a month |
| Sub-period Sharpe 2010–15 / 2016–20 / 2021–26 | 1.31 / 0.99 / 1.19 |
| Calendar years, long-short gross (%) | 2010 +33, 2011 +73, 2012 −4, 2013 +28, 2014 +36, 2015 +57, 2016 +53, 2017 +7, 2018 −17, 2019 −7, 2020 +226, 2021 +48, 2022 +23, 2023 +44, 2024 +54, 2025 +30, 2026 (to August) +82 |

## 3. Borrow at retail scale and the $3m short-leg rule

File: `borrow_variants_rf.csv`.

Schedules by daily dollar volume, assumed not quoted: base 1% above $25m, 4% at $10–25m, 12% at $3–10m, 35% below $3m; harsh 2 / 8 / 25 / 60%.

| Short-leg rule | Gross Sharpe | Average borrow | Net Sharpe (base / harsh) | Net 2021–26 | Shorts a month |
|---|---|---|---|---|---|
| All liquid names, flat 3% borrow | 1.19 | 3% | 0.96 | 1.00 | 12 |
| All liquid names, tiered | 1.19 | 16 / 29% | 0.60 / 0.24 | 0.68 / 0.36 | 12 |
| **Names above $3m a day (exported rule)** | **1.37** | **7 / 14%** | **1.02 / 0.81** | **1.18 / 1.00** | **9** |
| Names above $10m a day | 1.04 | 2 / 5% | 0.84 / 0.77 | 0.91 / 0.84 | 7 |
| Names above $25m a day | 0.50 | 1 / 2% | 0.35 / 0.33 | 0.39 / 0.36 | 6 |
| Long leg, short PGJ at 1% | 0.53 | 1% | 0.36 | −0.07 | |

## 4. The $500,000 account, 2010–26

Files: `bt500k_rf_variants.csv`, `bt500k_rf_by_year.csv`, `bt500k_rf_monthly.csv`; figures 2 to 4.

Daily P&L on the monthly holdings; dollars a side = min(80%, 10% ÷ σ) × equity with σ from the unit long-short's 60-day volatility blended with its long-run mean; 20 bp a side; borrow per name by volume; free cash at the bill less 0.5%, short proceeds above $100k at the bill less 1.25%; Reg T margin.

| | Random forest (headline) | Same account on the ensemble forecasts |
|---|---|---|
| Final equity from $500,000 | **$1.81m** | $1.36m |
| CAGR / volatility | **8.0% / 11.1%** | 6.2% / 10.9% |
| Sharpe, daily / monthly | **0.75 / 1.01** | 0.60 / 0.81 |
| Worst drawdown | −15.0% (Oct 2017 to Mar 2019, recovered Mar 2020) | −12.2% |
| Worst month / hit rate | −4.6% / 58% | −5.0% / 56% |
| Sub-period Sharpe 2010–15 / 2016–20 / 2021–26 | 0.55 / 0.76 / 0.99 | 0.32 / 0.64 / 0.92 |
| Exposure a side, average (range) | 24% (5 to 37%) | 24% |
| Margin pledged, average / peak (Reg T initial); maintenance peak; call days | 24% / 41%; 22%; 0 | |
| Commissions and spread / borrow (6.6% a year) / cash interest earned | $200k / $232k / +$209k | |
| Flat months (too few names) | 10 (Dec 2009, Jul 2010, Dec 2011, May 2012, Jun 2012, Jul 2012, Aug 2012, Sep 2012, Oct 2012, Nov 2012) | |

Calendar years, return on the account (%): 2010 +10.6, 2011 +14.2, 2012 −0.4, 2013 +3.0, 2014 +4.5, 2015 +5.2, 2016 +12.0, 2017 +1.8, 2018 −1.0, 2019 −5.5, 2020 +31.3, 2021 +10.6, 2022 +11.4, 2023 −2.5, 2024 +12.0, 2025 +12.7, 2026 (to August) +19.8. The best five months are 25% of the P&L, the best ten 47%.

Sizing (the same holdings, different exposure rules; all after costs, borrow and cash interest):

| Sizing | CAGR | Final equity | Volatility | Sharpe daily / monthly | Worst drawdown | Margin peak (Reg T initial) |
|---|---|---|---|---|---|---|
| **10% volatility target (headline)** | 8.0% | $1.81m | 11.1% | 0.75 / 1.01 | −15% | 41% |
| 15% volatility target | 11.4% | $3.02m | 16.7% | 0.73 / 0.96 | −23% | 62% |
| 20% volatility target | 14.7% | $4.91m | 22.4% | 0.72 / 0.94 | −29% | 85% |
| Fixed 50% a side | 16.5% | $6.45m | 24.8% | 0.74 / 1.02 | −23% | 73% |
| Fixed 80% a side | 25.3% | $21.61m | 40.6% | 0.76 / 0.99 | −36% | 130% |

Cost and borrow variants (Sharpe daily): harsh borrow 0.60; flat 8% borrow 0.72; no borrow 0.89; 10 bp costs 0.81; 40 bp costs 0.63; 60-day volatility without the long-run blend 0.72. Short-leg floor at $10m instead of $3m: $1.54m, CAGR 7.0%, Sharpe 0.75 / 0.85, drawdown −19%; at $25m: $863k, 0.40.

## 5. Monte Carlo and nulls

File: `bt500k_rf_mc.json`; figure 4.

Block bootstrap, 12-month blocks, 10,000 paths of 200 months: Sharpe median 1.01, 90% interval 0.62 to 1.40, above zero in every path; final equity $1.04m (5th percentile), $1.80m (median), $3.20m (95th); CAGR 4.5 to 11.8%; worst drawdown median −11%, one path in twenty worse than −19%, one in a hundred worse than −23%; one-year P&L −$25k / $33k / $134k at the 5th / 50th / 95th percentiles, 15% chance of a losing year; 1.5% chance of being down after five years. Beats the long leg alone in 95% of paired paths and the full-short-leg account in 100%. The bootstrap resamples this one history; it measures sampling noise around these sixteen years, not the chance that the next cycle differs.

Random-sort null: 500 random rankings through the same universe, construction, exposure path, costs and borrow give Sharpe −0.59 on average (sd 0.26), 95th percentile −0.15, maximum +0.11; none reaches the observed 1.01. The null sits below zero because a random version of the rule is long an average liquid name and short an average name above $3m a day, a drag the forecasts have to overcome first.

Stale forecasts (rank by the forecast made s months earlier; unit long-short, gross Sharpe): 1.37 at s = 0, 0.89 at 1, 0.92 at 2, 0.76 at 3, 0.41 at 6, 0.14 at 12.

Deflated Sharpe: with the dispersion of the 13 strategies examined on the research panel, the expected maximum Sharpe under a no-skill null is 0.79 for 13 trials and 1.11 for 70. The account's monthly Sharpe of 1.01 (skew 0.45, kurtosis 3.3) gives a deflated-Sharpe probability of 0.83 at 13 trials and 0.33 at 70. The random forest was the best of the eight models on this panel, so the 70-trial reading is the honest one.

## 6. The last five years as a fresh account

September 2021 to August 2026. Files: `bt500k_rf_5y_*`; figures 5 and 6, `bt500k_rf_5y_pages.pdf`.

Final equity $852k (+$352k, of which $83k is cash interest), CAGR 11.2%, volatility 11.3%, Sharpe 1.00 daily / 1.43 monthly, worst drawdown −8.8% (February 2025 to May 2025, recovered July 2025), worst month −3.4%, hit rate 70%; exposure 23% a side on average; borrow 6.5%; by year 2021 (Sep–Dec) +5.0%, 2022 +11.3%, 2023 −2.9%, 2024 +11.6%, 2025 +12.5%, 2026 (to August) +19.6%. Bootstrap Sharpe median 1.42, 90% interval 0.72 to 2.49; final equity $654k to $1.16m; worst drawdown median −5%, one in twenty −7%; 0.0% chance of being down after five years. Random-sort null on the 60 months: none of 300 random sorts reaches 1.43 (maximum 0.79). Deflated Sharpe 0.91 / 0.75. The rolling five-year Sharpe of the 2010-start account is 1.46 today against a median of 0.83 and a range of 0.32 to 1.46 over 141 windows. The long leg alone lost money over the five years (−0.9% a year): the result is the short leg.

## 7. Cross-checks and limits
- Price data: Yahoo and an independent archive agree on 99.95% of overlapping stock-months. Two data bugs were found and fixed in review (an asymmetric return filter that turned bad prints into −100% months, and a label-indexing error in the cleaning step that nulled unrelated days when sources were concatenated); the numbers here are from the corrected pipeline.
- Survivorship: names that delisted after April 2020, other than twelve SOEs carried through Hong Kong, are missing for their final months. The volume-weighted universe correlates 0.92 with PGJ and trails it by 4% a year in 2021–26, so the survivor premium is not driving the recent numbers; the small-cap delistings sit in the short leg and remain unmeasured.
- Concentration: the pool above $3m a day was 14 to 58 names a month until 2017 and 43 to 108 since, so the short leg has held 5 to 11 names before 2018 and 8 to 21 since; at the minimum of five names a single short is 20% of the leg.
