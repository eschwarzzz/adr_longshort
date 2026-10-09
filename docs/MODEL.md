# Model description: Chinese ADR random-forest long-short

## What it is
An application of a cross-sectional forecast of next-month stock returns from "Empirical Asset Pricing via Machine Learning" (Gu, Kelly, and Xiu, 2020), applied to the US tradeable Chinese market. This strategy has exploitable alpha primarily due to high borrow costs of rebalancing so many short positions monthly forcing a capital constraint of sub $1M. Further evidence for this being the case is that the same exact strategy run over US equities returns a Sharpe of 0.13, demonstrating the decay of alpha within the US domestic market, as this basic mechanism is undoubtedly used to better effect domestically by various long-short funds, which cannot enter our market due to the liquidity constraint being too tight. The forecasts are turned into an equal-weight long-short portfolio: long the top fifth of the liquid universe, short the bottom fifth of the names that trade at least $10m a day. Nothing in the model uses fundamentals, analysts, news or flows.

Headline result: on a $500,000 account with a 10% volatility target, after commissions, borrow and cash interest, the strategy grows to $1.74m over January 2010 to August 2026, a 7.7% compound annual return at 10.8% volatility, Sharpe 0.75 on daily returns (0.86 monthly), worst drawdown −19.3%, with the money arriving in bursts (2010–11, 2013, 2016–17, 2020–21, 2025–26) around flat years. Before sizing, the dollar-neutral portfolio earns 36% a year gross, Sharpe 1.02, of which the short leg contributes −19% a year and the long leg +17%; it is close to beta-neutral as well, with a beta of −0.15 and a correlation of −0.13 to the PGJ China ETF (long leg beta 0.95, short leg 1.09).

## Universe
US-listed Chinese companies (ADRs and direct listings) assembled from the USCC lists of 2019, 2020 and 2025 and the Nasdaq screener: 453 names over 2000–2026, 302 still listed. Prices are dividend- and split-adjusted daily closes with volume. Twelve SOEs that delisted in 2021–23 are carried to their delisting dates through their Hong Kong lines. `results/predictions_rf.csv` lists every ticker-month the model has scored.

## Inputs
Daily prices are cleaned first (`src/clean_prices.py`): isolated prints that revert the next day are dropped, both days of a crash-then-rebound pair are removed, and any remaining single day above +300% (an unadjusted split) is removed. Then at each month-end t, from that name's own history up to t (`src/features.py`):

| Characteristic | Definition |
|---|---|
| ret_1m | return over month t |
| mom_6_1, mom_12_1 | cumulative return over months t−5..t−1 and t−11..t−1 |
| mom_36_13 | cumulative return over months t−35..t−12 |
| vol_1m, vol_3m | standard deviation of daily returns in month t; average of the last three months' |
| ivol_12m, beta_12m | residual volatility and beta against the equal-weight universe over the trailing 12 months of daily returns |
| maxret_1m, minret_1m | largest and smallest daily return in month t |
| dolvol_1m | log of average daily dollar volume in month t |
| dolvol_trend | dolvol_1m minus its trailing 12-month mean |
| illiq_1m | log Amihud ratio: mean of abs(return) / dollar volume, × 1e6 |
| zero_1m | share of zero-return days in month t |
| high52 | month-end price divided by the 52-week high |
| age | log of months since first price |

Each characteristic is converted to its percentile rank within that month's tradeable cross-section (price ≥ $1, ≥ $100k a day) and mapped to [−1, 1]; a missing value becomes 0, the median. The model sees only these sixteen ranks, in the order given in `model/feature_spec.json`. Ranks must be computed over the same kind of universe the model was fitted on: feeding it ranks from a different universe (say, all US stocks) changes what "+1 on dolvol_1m" means.

## Target
The next calendar month's return minus the 1-month T-bill. For fitting only, the target is winsorised at the training sample's 1st and 99th percentiles (about −0.45 and +0.71 for the latest model).

## Estimator and refit protocol
scikit-learn `RandomForestRegressor(n_estimators=300, max_features=0.5, random_state=0)`, with `max_depth` ∈ {2, 4, 6} and `min_samples_leaf` ∈ {20, 50} chosen by validation mean squared error. The protocol is an annual refit: in December before test year Y, train on 2001..Y−3, validate on Y−2..Y−1, pick the six-candidate winner, and use it unchanged for the twelve months of Y. No test-year data touches any fit. The hyperparameters chosen each year are in `model/rf_hyperparams_by_year.csv`: shallow trees (depth 2) in most years to 2019 and again for 2026, depth 4 to 6 in 2016–17 and 2020–25.

## What the forest has learned
Impurity-based importance for `rf_2026` (`model/feature_importance.csv`): zero_1m 0.33, high52 0.23, ivol_12m 0.08, mom_6_1 0.05, dolvol_trend 0.05, minret_1m 0.05, vol_3m 0.05, ret_1m 0.04, everything else below 0.04. The first thing the forest does is separate names with many zero-return days (thin, stale, often halted or about to be) from the rest, then it reads price relative to the 52-week high, idiosyncratic volatility and recent returns. The linear models fitted alongside it in the research agree on the direction of the stable terms: liquid, calm names near their highs score high; thin, volatile, recently-collapsed names score low. In the quintile sorts the short leg carries the result: the bottom quintile of liquid names lost about 27% a year over 2010–26, the top quintile earned about 14% a year, roughly what a volume-weighted index of the universe earned.

Forecast quality out of sample, 2010–26, all tradeable names: R² against a zero forecast 0.35%, rank IC 0.080 (t 7.0), 0.081 on the liquid names (t 6.1).

## From forecasts to positions (`src/portfolio.py`)
At each month-end, among tradeable names with ≥ $1m a day (25 or more required, else flat): long the top quintile by forecast, equal weight. Among those also trading ≥ $10m a day (10 or more required, else flat): short the bottom quintile, equal weight, at least five names. Hold one calendar month, buy-and-hold inside it, re-form at the next month-end. The $10m floor is applied only here; the model is fitted on the whole tradeable panel. Its purpose is borrow: the thin third of the unrestricted short leg costs more to borrow at retail than it earns, and removing it leaves the spread intact.

Sizing in the $500k backtest: dollars a side = min(80%, 10% ÷ σ) × equity, resized monthly, σ = 0.7 × trailing 60-day realised volatility of the one-dollar-a-side long-short + 0.3 × its expanding mean. Fixed exposure (50% a side) earns about twice as much at one and a half times the drawdown and the same Sharpe; the volatility target cut exposure to 15–20% a side in 2025–26 when the unit volatility ran at 50–70%.
