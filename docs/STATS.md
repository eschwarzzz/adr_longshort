# Backtest statistics: random-forest Chinese ADR long-short

All numbers are out of sample in the sense that no model saw data from its test year; the design (features, universe, model menu, the $10m floor) was fixed with the research and its literature in view. Window January 2010 to August 2026. Generated from the files in `results/` by `docs/make_stats.py`.

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
| Calendar years, long-short gross (%) | 2010 +33, 2011 +73, 2012 −4, 2013 +28, 2014 +36, 2015 +57, 2016 +53, 2017 +7, 2018 −17, 2019 −7, 2020 +226, 2021 +48, 2022 +23, 2023 +44, 2024 +54, 2025 +30, 2026 +82 (2026 to August) |

## 3. Borrow at retail scale and the $10m short-leg rule

File: `borrow_variants_rf.csv`.

Schedules by daily dollar volume, assumed not quoted: base 1% above $25m, 4% at $10–25m, 12% at $3–10m, 35% below $3m; harsh 2 / 8 / 25 / 60%.

| Short-leg rule | Gross Sharpe | Average borrow | Net Sharpe (base / harsh) | Net 2021–26 | Shorts a month |
|---|---|---|---|---|---|
| All liquid names, flat 3% borrow | 1.19 | 3% | 0.96 | 1.00 | 12 |
| All liquid names, tiered | 1.19 | 16 / 29% | 0.60 / 0.24 | 0.68 / 0.36 | 12 |
| Names above $3m a day | 1.37 | 7 / 14% | 1.02 / 0.81 | 1.18 / 1.00 | 9 |
| **Names above $10m a day (exported rule)** | **1.04** | **2 / 5%** | **0.84 / 0.77** | **0.91 / 0.84** | **7** |
| Names above $25m a day | 0.50 | 1 / 2% | 0.35 / 0.33 | 0.39 / 0.36 | 6 |
| Long leg, short PGJ at 1% | 0.53 | 1% | 0.36 | −0.07 | |

## 4. The $500,000 account, 2010–26

Files: `bt500k_rf_variants.csv`, `bt500k_rf_by_year.csv`, `bt500k_rf_monthly.csv`; figures 2 to 4.

Daily P&L on the monthly holdings; dollars a side = min(80%, 10% ÷ σ) × equity with σ from the unit long-short's 60-day volatility blended with its long-run mean; 20 bp a side; borrow per name by volume; free cash at the bill less 0.5%, short proceeds above $100k at the bill less 1.25%; Reg T margin.

| | Random forest (headline) | Same account on the ensemble forecasts |
|---|---|---|
| Final equity from $500,000 | **$1.74m** | $1.56m |
| CAGR / volatility | **7.7% / 10.8%** | 7.1% / 10.7% |
| Sharpe, daily / monthly | **0.75 / 0.86** | 0.70 / 0.76 |
| Worst drawdown | −19.3% (Nov 2017 to Mar 2019, recovered Jun 2020) | −16.6% |
| Worst month / hit rate | −8.4% / 57% | −8.3% / 55% |
| Sub-period Sharpe 2010–15 / 2016–20 / 2021–26 | 0.78 / 0.61 / 0.86 | 0.57 / 0.63 / 0.87 |
| Exposure a side, average (range) | 28% (12 to 44%) | 28% |
| Margin pledged, average / peak (Reg T initial); maintenance peak; call days | 28% / 49%; 27%; 0 | |
| Commissions and spread / borrow (2.3% a year) / cash interest earned | $224k / $95k / +$211k | |
| Flat months (too few names) | 11 (Dec 2009, Jul 2010, Dec 2011, May 2012, Jun 2012, Jul 2012, Aug 2012, Sep 2012, Oct 2012, Nov 2012, Feb 2013) | |

Calendar years, return on the account (%): 2010 +13.6, 2011 +15.4, 2012 −0.6, 2013 +11.6, 2014 +4.6, 2015 −0.7, 2016 +18.8, 2017 +11.0, 2018 −6.4, 2019 −10.6, 2020 +25.1, 2021 +10.7, 2022 +9.6, 2023 +1.1, 2024 −0.8, 2025 +10.4, 2026 +24.8 (2026 to August). The best five months are 28% of the P&L, the best ten 50%.

Sizing and cost variants (final equity, CAGR, Sharpe daily, worst drawdown): 15% vol target $2.80m, 10.9%, 0.72, −28%; 20% target $4.38m, 13.9%, 0.71, −37%; fixed 50% a side $4.18m, 13.5%, 0.74, −31%; fixed 80% a side $10.61m, 20.1%, 0.73, −46% with 0 margin-call days; harsh borrow 0.69; flat 8% borrow 0.60; no borrow 0.81; 10 bp costs 0.82; 40 bp costs 0.62.

Other legs on the same account: full short leg at retail borrow (16% a year) $1.13m, Sharpe 0.38; short leg above $25m $815k, 0.34; long leg alone $987k, 0.48; long leg hedged with PGJ $794k, 0.34.

## 5. Monte Carlo and nulls

File: `bt500k_rf_mc.json`; figure 4.

Block bootstrap, 12-month blocks, 10,000 paths of 200 months: Sharpe median 0.87, 90% interval 0.47 to 1.28, above zero in 100% of paths; final equity $950k (5th percentile), $1.74m (median), $3.18m (95th); CAGR 3.9 to 11.7%; worst drawdown median −16%, one path in twenty worse than −25%, one in a hundred worse than −31%; one-year P&L −$43k / $44k / $135k at the 5th / 50th / 95th percentiles, 19% chance of a losing year; 3.0% chance of being down after five years. Beats the long leg alone in 88% of paired paths and the full-short-leg account in 94%. The bootstrap resamples this one history; it measures sampling noise around these sixteen years, not the chance that the next cycle differs.

Random-sort null: 500 random rankings through the same universe, construction, exposure path, costs and borrow give Sharpe −0.55 on average (sd 0.25), 95th percentile −0.11, maximum +0.09; none reaches the observed 0.86. The null sits below zero because a random version of the rule is long an average liquid name and short an average name above $10m a day, a drag the forecasts have to overcome first.

Stale forecasts (rank by the forecast made s months earlier; unit long-short, gross Sharpe): 1.04 at s = 0, 0.85 at 1, 0.67 at 2, 0.58 at 3, 0.05 at 6, −0.03 at 12.

Deflated Sharpe: with the dispersion of the 13 strategies examined on the research panel, the expected maximum Sharpe under a no-skill null is 0.79 for 13 trials and 1.11 for 70. The account's monthly Sharpe of 0.86 (skew −0.02, kurtosis 3.5) gives a deflated-Sharpe probability of 0.62 at 13 trials and 0.16 at 70. The random forest was the best of the eight models on this panel, so the 70-trial reading is the honest one.

## 6. The last five years as a fresh account

September 2021 to August 2026. Files: `bt500k_rf_5y_*`; figures 5 and 6, `bt500k_rf_5y_pages.pdf`.

Final equity $804k (+$304k, of which $80k is cash interest), CAGR 9.9%, volatility 11.3%, Sharpe 0.89 daily / 1.22 monthly, worst drawdown −13.0% (April 2024 to November 2024, recovered January 2026), worst month −4.9%, hit rate 63%; exposure 24% a side on average; borrow 2.4%; by year 2021 (Sep–Dec) +7.3%, 2022 +9.5%, 2023 +0.7%, 2024 −1.2%, 2025 +10.2%, 2026 (to August) +24.7%. Bootstrap Sharpe median 1.22, 90% interval 0.30 to 2.53; final equity $555k to $1.21m; worst drawdown median −8%, one in twenty −14%; 1.1% chance of being down after five years. Random-sort null on the 60 months: none of 300 random sorts reaches 1.22 (maximum 0.52). Deflated Sharpe 0.81 / 0.59. The rolling five-year Sharpe of the 2010-start account is 1.25 today against a median of 0.73 and a range of 0.24 to 1.25 over 141 windows. The long leg alone lost money over the five years (−1.3% a year): the result is the short leg.

## 7. Cross-checks and limits
- Price data: Yahoo and an independent archive agree on 99.95% of overlapping stock-months. Two data bugs were found and fixed in review (an asymmetric return filter that turned bad prints into −100% months, and a label-indexing error in the cleaning step that nulled unrelated days when sources were concatenated); the numbers here are from the corrected pipeline.
- Survivorship: names that delisted after April 2020, other than twelve SOEs carried through Hong Kong, are missing for their final months. The volume-weighted universe correlates 0.92 with PGJ and trails it by 4% a year in 2021–26, so the survivor premium is not driving the recent numbers; the small-cap delistings sit in the short leg and remain unmeasured.
- Concentration: the pool above $10m a day was 10 to 25 names until 2017, so the short leg was its minimum of five names; it has been 7 to 12 since 2018.
