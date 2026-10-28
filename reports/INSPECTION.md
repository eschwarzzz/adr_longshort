# Chinese ADR long-short: inspection notes

**Version 1, 19 October 2026.** These notes cover the repository as cloned on 18 October 2026: reproducibility, the data and universe (look-ahead, liquidity, survivorship, cleaning), the model and its refit protocol, the features, the portfolio construction, costs and borrow, portability to other markets, and a pre-specified final candidate. Nothing is committed; everything is local under `research/` (scripts) and `research/out/` (tables, logs).

Unless marked otherwise, figures run January 2010 to August 2026. "Unit book" means $1 long and $1 short, monthly, from the forecast file. "Account" means the repository's $500,000 simulation: 10% volatility target, 80% cap, 20 bp per side, the tiered borrow schedule, cash interest and Reg T margin (`backtest.py`). Sharpe ratios are daily unless marked "monthly". "Shipped" means the repository as cloned. "Corrected" means the data fixes of section 2.

## Summary

1. **The pipeline reproduces.** The tests pass (6 of 6), and `refit.py --check` matches the shipped forecasts to 1e-16.
2. **The universe used future information, and fixing it costs performance.** Five fixes, each measured on its own:
   - **Price test on actual traded prices.** The $1 test used split-adjusted prices. The rule dropping 55 serial reverse-splitters for their whole history used their future price path.
   - **No future-data filter.** The scored pool required ten trading days of next-month data.
   - **Median volume.** The $1m and $3m floors used the monthly *mean* dollar volume, which one pump day can inflate.
   - **Ticker reuse.** 25 names carry an earlier company's price history under the same ticker; CIH, NEW, WEI and HMI go back to 2000.
   - **Result:** the account falls from $1.81m (Sharpe 0.75, monthly 1.01) to **$1.34m (0.62, monthly 0.69)**. Median volume is the largest single fix: it cuts the short leg from −28% to −10% a year on its own.
3. **The cleaning rules do not flatter the P&L.** The audit verified that 777 of the 871 removals were data errors. Restoring the real and unverifiable moves changes the account by less than $10k. The +300% rule is asymmetric (it deletes the up-day of a squeeze that doesn't reverse), but no such day falls inside a corrected-book position.
4. **The short-side tail is real, but it is not a margin problem at this sizing.**
   - **History:** in November 2020 (the Chinese EV mania) the short book rose 55% in a month, and the corrected account lost 12.2%.
   - **Shock test:** a +200% day on the largest short costs 7% of equity in a typical month, up to 13%, and 16% at the five-name minimum.
   - **Margin:** Reg T maintenance never binds, because about 75% of equity is cash.
5. **The survivorship gap is large and can't be signed with free data.**
   - **Missing:** 266 China- and HK-based issuers delisted from US exchanges in 2009–26, and only 17 have prices through delisting. In 2011–13 the true short pool was plausibly 1.5–3× the backtested one.
   - **Direction:** fraud collapses would have helped on paper but were often halted; take-privates at a premium would have hurt.
6. **Freezing the model for two years is not a problem.** Every way of training on more recent data does worse; retraining through Y−1 loses at t −2.3. Keep the freeze.
7. **The forest beats Huber and OLS where the money is, not on average.**
   - **Average:** Huber's rank IC is higher (0.12 vs 0.09).
   - **Tails:** the forest's book is far better (account $1.61m vs $0.74m), because its bottom quintile among the $3m names falls 18.7% a year against Huber's 9.3%.
   - **Significance:** forest minus Huber has paired t 2.1, about 2 across seeds. Choosing the model each December by validation is the worst option of all.
8. **No feature change survives.**
   - **zero_1m:** it ranks first by the forest's own importance but has no signal inside the traded sets (rank IC −0.007 at ≥ $3m). Dropping it is neutral on Sharpe once selection happens inside the protocol.
   - **The redundant five:** illiq_1m, maxret_1m and minret_1m are 80–92% explained by the other features. Dropping all five looked neutral or slightly positive on one corrected panel, but scored lower (0.50 vs 0.62) when pre-registered on the final one.
9. **The short leg needs a hedge, and the long leg is the best one over the full period.**
   - **Short only:** account Sharpe 0.21 with a −31% drawdown.
   - **Long-leg alpha:** the long leg earns about 11% a year of alpha against the volume-weighted universe, almost none of it net since 2021.
   - **Other constructions lose alpha:** ETF hedges, beta-matched hedges, inverse-volatility shorts and borrow-aware ranking (paired t −1.3 to −3.3).
   - **The 2021 hedge break:** the "micro-IPO" explanation is mostly rejected. The long leg's stock-picking edge faded.
10. **Borrow is the binding constraint.**
    - **Observed fees:** where IBKR fees are observed (October 2025 to October 2026), the model's shorts cost a median 58–67% a year, against the 6.4% assumed.
    - **Imputed history:** a fee model fitted to those observations puts the 2010–26 average at about 37%.
    - **Hard-to-borrow concentration:** names that fail a pre-stated availability rule carry 40–50% of the short leg's gross P&L, rising to 88% in 2025–26.
11. **Under realistic frictions the strategy is close to cash.**
    - **Assumptions:** per-name trading costs (half to all of the high-low spread estimate, plus a tick floor, commission and impact) and observed-or-imputed IBKR fees.
    - **Result:** the fully corrected account ends at **$0.52m–$0.69m (Sharpe 0.07–0.24)**. Borrow-aware ranking cuts the fee to about 4% a year but loses the alpha with it.
    - **Diagnosis:** the alpha *is* the hard-to-borrow premium.
12. **The edge exists elsewhere, but only China's version pays.**
    - **Other markets:** US microcaps, HKEX small caps and non-China Asian micro-listings all show positive rank IC and a lagging bottom quintile, but only the China book is profitable net. Elsewhere the long-short carries heavy short beta.
    - **Hong Kong:** the bottom quintile's underperformance sits almost entirely in names that can't be shorted (−22% to −26% a year against roughly 0). That supports a short-constraint story, with a caveat: the flag is today's list, not a historical one.
13. **Evidence of skill does not survive the trial count.**
    - **Random sorts:** none of 500 match the corrected account (monthly Sharpe 0.69).
    - **Deflated Sharpe:** 0.36 at 13 trials, 0.05 at the repository's own 70, and 0.005 at 322 (the repository's 70 plus the 252 logged here, a lower bound). For the shipped account the same figures are 0.83, 0.33 and 0.07.

---

## 1. What the strategy is, and what goes into the model

At each month-end, 16 price-and-volume characteristics are computed for every name. Each is ranked within that month's tradeable cross-section (price ≥ $1, ≥ $100k a day) and mapped to [−1, 1], with missing values set to 0.

| Group | Characteristics |
|---|---|
| Trend | ret_1m (reversal), mom_6_1, mom_12_1 (momentum), mom_36_13 (long-term reversal) |
| Risk | vol_1m, vol_3m, ivol_12m, beta_12m, maxret_1m, minret_1m |
| Liquidity | dolvol_1m, dolvol_trend, illiq_1m, zero_1m |
| Other | high52, age |

The four groups are labels only; the forest sees 16 inputs.

**Target and refit protocol.**
- **Target:** the next month's excess return, winsorised at the 1st and 99th percentiles.
- **Search:** each December, six random forests are fitted (depth 2, 4 or 6 × leaf size 20 or 50) on 2001 to Y−3. The one with the lowest validation MSE on Y−2 to Y−1 trades year Y unchanged. It is not refitted on the validation years (`adr_longshort/rf_model.py`).

**The portfolio** (`adr_longshort/portfolio.py`):
- **Long:** the top quintile of names trading ≥ $1m a day.
- **Short:** the bottom quintile of names trading ≥ $3m a day, at least five names.
- **Weighting:** equal weight, held one month.

The "about 60 names" in the summary is not a parameter. It is however many names clear $1m in a given month: 62 on average on the shipped data, 58 corrected. The code only needs 25.

The short leg is not "the most liquid of the most illiquid." It is the worst-forecast fifth of the names liquid enough to borrow. The model is fitted on the wide ≥ $100k universe; the floors apply only at portfolio formation.

## 2. The data and the universe

Reproduce with:
- `python research/p0_splits.py` (fetches splits)
- `python research/p0_prices.py`
- `python research/p0_baseline.py`
- `python research/p0_accounts.py`
- `python research/p0_reuse.py`

### 2.1 Four look-ahead and liquidity fixes, each measured on its own

| | Shipped | Real-price test, no $5k rule | No next-month filter | Median volume | All four |
|---|---|---|---|---|---|
| Unit long-short Sharpe | 1.34 | 1.24 | 1.32 | **1.01** | 1.11 |
| Short leg, %/yr | −28.5 | −29.7 | −28.8 | **−10.4** | −20.1 |
| Account Sharpe | 0.75 | 0.69 | 0.75 | 0.56 | 0.68 |
| Account 2021–26 | 0.98 | 0.83 | 0.98 | 0.50 | 0.68 |
| Worst month | −4.6% | −8.5% | −4.3% | −7.4% | −12.2% |
| Final equity | $1.79m | $1.67m | $1.80m | $1.23m | $1.61m |

- **Real prices.**
  - **Splits:** 360 yfinance split events (333 reverse) undo the split adjustment. 346 were absorbed correctly in Yahoo's adjusted series, 9 never were, and GMM's November 2024 1-for-15 split is missing from Yahoo's list entirely.
  - **The $5,000 rule:** on traded prices only 3 names ever exceed $5,000 (all bad-data names), against 61 on adjusted prices. The rule removed 55 names that all lost 99–100%. That is the short leg's natural population, removed using future prices.
- **The next-month filter** turns out to guard mostly against the Kaggle archive ending on 1 April 2020, not against halts. Only 27 rows are affected, 21 of them in March 2020. The fix scores every tradeable row except archive truncations.
- **Median volume.**
  - **Mean inflation:** 13% of the shipped short book's name-months (21% since 2021) cleared $3m only on the mean. On a typical day they traded under $1.5m, e.g. GMM, which qualified on one $70m day.
  - **Concentration:** those names fell 10.6% a month, against 2.5% for the rest.

### 2.2 Ticker reuse

25 names have prices starting more than a year before their IPO year. Some are genuine over-the-counter or reverse-merger histories (CREG, KNDI, CPHI). Others belong to an earlier company with the same ticker: CIH, NEW, WEI and HMI from 2000, plus GSX, CTK, SVA and WSG.

The worst account day in the backtest (18 July 2011, −7.8%) was "NEW", a different company's +150% day, in the short book.

Dropping each name's prices before January of its IPO year minus one gives the fully corrected baseline:

| | Shipped | Fully corrected |
|---|---|---|
| Final equity | $1.81m | $1.34m |
| Sharpe, daily / monthly | 0.75 / 1.01 | 0.62 / 0.69 |
| Worst drawdown | −15.0% | −14.7% |
| Worst month | −4.6% | −9.6% |
| Unit Sharpe 2010–15 / 2016–20 / 2021–26 | 1.40 / 1.18 / 1.50 | 1.13 / 0.88 / 0.96 |
| Account Sharpe 2021–26 | 0.98 | 0.79 |
| Five-year fresh account, Sep 2021 – Aug 2026 | $852k | $776k (Sharpe 0.83) |

### 2.3 Survivorship

Reproduce with `research/p07_*.py`.

**The data contains no company that died before April 2020**, and none that listed after April 2020 and has since delisted:
- The Kaggle archive is a snapshot of names listed in April 2020, with histories to that date.
- Yahoo carries only names listed today.
- `data/README.md` describes the Kaggle names as "delisted before April 2020". They are in fact names that delisted *after* it, missing their final months.

SEC Form 25 filings count what is missing:
- **266 operating China/HK issuers** finally delisted in 2009–26: 174 before April 2020 and 92 after. Only 17 have prices through delisting.
- **The 2011–12 reverse-merger fraud wave** (35 exchange-initiated delistings in 2011–12) is absent entirely.
- **The micro-IPO graveyard** (IPO after April 2020, delisted by October 2026): 21 names, 2 with any prices.

The bias can't be signed:
- **Fraud collapses** would have been gains for the short leg on paper, but were often halted for months with borrow accruing.
- **Take-privates** (123, clustered in 2012–16 and 2020–21) pay 20–40% premiums on exactly the beaten-down names the model shorts.

The 2010–15 results are the least reliable.

## 3. Cleaning, audited

Reproduce with `python research/p0_audit.py`.

The shipped cleaning removes three things: isolated prints that revert the next day (rows dropped), both days of a crash-then-rebound pair, and any remaining day above +300%.

Each removal was logged with ticker, date, raw return, filter, side and position size. It was then classified by rules fixed before looking at P&L:
- a yfinance split nearby
- an unlisted-split signature (the level persists, volume moves the opposite way)
- mixed price scales (the price flips by three times or more and back)
- Nasdaq's historical close as a second source
- whether the close sits inside the day's own high-low range
- volume confirmation

| Verdict | Count | Mainly |
|---|---|---|
| Verified error | 777 | mixed price scales 687, unlisted splits 45, second source disagrees 36, listed splits 9 |
| Real move | 46 | Nasdaq confirms 38, volume confirms 8 |
| Unknown | 48 | archive names without a second source |

**Only 12 removals fell inside a shipped-book position, and none inside a corrected one.**
- CIH (Kaggle) price-scale flips, all in long positions.
- AIXI on 29 July 2026: Yahoo shows +693%, Nasdaq +13%. A Yahoo error.
- **STAK on 24 July 2026: +602%, confirmed by Nasdaq.** A real squeeze, in a short.

**Asymmetry.** The few removals that touched the book did lean the strategy's way. Shorts lost 2 up-days; longs lost 8 down-days and 2 up-days. But 10 of those 12 were verified errors.

**Restored P&L.** Only verified errors stay out; real and unknown moves are put back.

| Book | Shipped cleaning | Only verified errors removed | No cleaning |
|---|---|---|---|
| Shipped | $1.806m, 0.75 | $1.798m, 0.75, worst month −4.57% | $1.742m, 0.72 |
| Corrected | $1.608m, 0.68 | identical | identical |

No margin call on any of them.

**Phase 5 found one asymmetry the audit's sample didn't contain.** When a squeeze doesn't reverse the next day, the +300% rule deletes the up-day but keeps the fall that follows. AMTD Digital (HKD, not in the universe) shows September 2022 as −90.5% instead of the real −58%.

The fix is the one used for the final candidate: training may use a filter chosen on validation, but P&L removes only verified errors.

## 4. Short-side stress

Reproduce with `python research/p0_stress.py`.

**Intraday spikes.** A short rose more than 100% intraday in 27 months of the shipped book and 12 months of the corrected one. STAK reached +809%.
- **Worst case:** with every short at its high on the same day, the account loses at most 9.1–9.4%, and maintenance stays at or below 22% of equity.

**The +200% shock** (the largest short triples on one day mid-month):

| | Typical month | Worst month | At the five-name minimum |
|---|---|---|---|
| Account loss | 6–7% | 13% | up to 16% |
| Margin calls | 0 | 0 | 0 |

**Historical tail.**
- **November 2020, the EV mania:** the corrected short book rose 55% (CBAT +228%, KXIN +104%, LI +57%, KNDI +45%), and the account lost 12.2%.
- **AMTD Digital:** its $1,679 peak in August 2022 came before the IPO-month rule allowed a position. Once added to the universe, it was shorted in 11 months. September 2022 saw a +312% day, with the position reaching 1.69× entry at the worst close.

**Verdict.** At 24% exposure the danger is P&L and broker buy-ins or recalls, not Reg T. Monthly buy-and-hold has no stop. A single real squeeze costs about three times the worst month the backtest shows.

## 5. Model and protocol

Reproduce with `research/p1_*.py`.

### 5.1 zero_1m and the traded universe

Rank IC, 2010–26, median-volume sets:

| Feature | ≥ $100k | ≥ $1m | ≥ $3m |
|---|---|---|---|
| zero_1m | −0.021 (t −2.2) | −0.023 (−2.0) | −0.007 (−0.5) |
| high52 | 0.101 (7.7) | 0.077 (4.4) | 0.070 (3.5) |
| ivol_12m | −0.121 (−9.0) | −0.086 (−4.9) | −0.071 (−3.6) |

**Why the importance ranking misleads.**
- **Mostly zeros:** zero_1m is exactly zero in 63% of ≥ $100k rows, 76% at ≥ $1m and 81% at ≥ $3m.
- **Impurity vs permutation:** impurity importance still ranks it first even for a forest fitted only on ≥ $1m rows, but its permutation importance on liquid validation rows is negative.
- **Diagnosis:** the 0.33 importance is an artifact of impurity importance on a feature that splits off a thin tail of stale names. It describes the fitting universe, not the traded one.

### 5.2 The two-year freeze

Every candidate is a complete procedure run on data ending in Y−1. Selection uses each candidate's trailing 24 out-of-sample months, so no candidate is judged on rows it trained on.

| Training scheme | Unit Sharpe | Account Sharpe | Final | t vs freeze |
|---|---|---|---|---|
| Shipped freeze (train to Y−3) | 1.13 | 0.68 | $1.61m | — |
| One-year validation (train to Y−2) | 0.85 | 0.38 | $0.88m | −2.04 |
| Retrain through Y−1 | 0.77 | 0.34 | $0.85m | −2.33 |
| Rolling 6 / 8 years | 1.04 / 1.12 | 0.56 / 0.61 | | −0.94 / −0.51 |
| Recency-weighted | 1.23 | 0.72 | $1.70m | +0.10 |
| Quarterly refit | 0.93 | 0.48 | $1.07m | −1.36 |
| Selected each December (trailing long-short / IC) | 1.16 / 1.06 | 0.65 / 0.57 | | −0.34 / −1.06 |

**The freeze is a design choice, not a crack.** The recent years are the noisiest (validation error doubles after 2017), and adding them adds noise faster than information. Keep it and document why.

### 5.3 Forest, Huber, OLS, blends

Same sizing, costs and floor for all.

| Model | Rank IC | Unit Sharpe | Account Sharpe | Final | Top − bottom quintile, ≥ $1m / ≥ $3m (%/yr) | Bottom quintile at ≥ $3m |
|---|---|---|---|---|---|---|
| Random forest | 0.09 | 1.13 | 0.68 | $1.61m | 34.3 / 40.6 | −18.7% |
| Huber | 0.12 | 0.60 | 0.27 | $0.74m | 24.0 / 21.9 | −9.3% |
| OLS | 0.09 | 0.73 | 0.38 | $0.88m | 27.0 / 28.0 | −6.6% |
| Blend RF + Huber | 0.11 | 1.00 | 0.56 | $1.28m | | |
| Blend RF + OLS | 0.10 | 1.02 | 0.63 | $1.38m | 34.1 / 36.0 | −13.9% |
| Model chosen each December by validation | 0.10 | 0.77 | 0.40 | $0.95m | | |

**Paired tests on monthly net returns.**
- **Forest vs linear models:** forest minus Huber is +16.6% a year (t 2.12), and forest minus OLS t 2.19. Across seeds both are about 2.0 ± 0.45.
- **Forest vs blends:** not significant (t 0.45–0.78).

**Conclusions.**
- **Why the forest wins:** it is worse on average but better at the extremes it trades, which is the threshold effect. Its lead is borderline significant.
- **Choosing the model inside the protocol fails:** the choice flips almost every year.
- **The honest specification:** fix the forest in advance and drop the "linear beats forest" claim.

On the shipped panel OLS came close to the forest (1.31 vs 1.37). That depended on the uncorrected universe.

## 6. Features

Reproduce with `research/p2_*.py`. The first-round ablation on the shipped panel is in `research/features_ablation.py`.

### 6.1 Build-up

Hyperparameters are fixed per year, so smaller sets are disadvantaged.

| Feature set | Unit Sharpe | Account Sharpe | Final | Worst month |
|---|---|---|---|---|
| high52 + mom_6_1 + mom_12_1 + mom_36_13 | 1.33 | 0.76 | $1.78m | −8.0% |
| + zero_1m | 0.93 | 0.52 | $1.16m | −10.3% |
| + each remaining feature in turn | 1.06–1.36 | 0.57–0.76 | | |
| all 16 | 1.19 | 0.75 | $1.82m | |

**The four-feature trend forest matches the full 16** (paired t ≈ 0). Beta to PGJ is about 0 in every variant.
- **Add-one** (base plus one feature): only maxret_1m, ret_1m and vol_1m hold the base level. zero_1m, beta_12m and dolvol_1m hurt most.
- **Column order alone** moves Sharpe by about 0.06.

### 6.2 Pairs

- **Singles:** high52 0.98 and dolvol_trend 0.64 are the only strong ones.
- **Features that only work together:**
  - ivol_12m + mom_6_1: 0.88, from singles of 0.13 and 0.11.
  - ivol_12m + dolvol_trend: 1.04.
  - mom_6_1 + zero_1m: 0.57.

ivol_12m is useless alone and strong in combination. That's why the "calm, near the high" description reads like trend following without being only that.

Correlation of the strategy with trend-following funds is 0.06–0.08, not significant (`results/trend_correlation.csv`).

### 6.3 Dropping zero_1m, inside the protocol

Seeds 0–4; hyperparameters re-chosen each year.

| Variant | Unit | Account | Drawdown | Worst month | Account 2021–26 | Beta to VW universe |
|---|---|---|---|---|---|---|
| all 16 | 1.12 ± .12 | 0.68 ± .08 | −17.3% | −12.2% | 0.68 | −0.13 |
| without zero_1m | 1.12 ± .04 | 0.68 ± .03 | −15.4% | −6.4% | 0.48 | +0.01 |
| decided each December by validation | 1.11–1.14 | 0.66–0.67 | −15.6% | −6.4 to −12.2% | 0.61–0.62 | ~0 |

The shipped-panel gain (1.69 vs 1.37) does not survive the corrected universe.
- **What dropping it buys:** a smaller worst month and full beta neutrality.
- **What it costs:** 2021–26.
- **Verdict:** a risk trade-off, not an improvement. Not adopted.

### 6.4 The "redundant five"

How much of each feature's rank is explained by the other 15 (R²):

| Feature | R² | Most correlated with |
|---|---|---|
| illiq_1m | 0.92 | dolvol_1m (−0.91) |
| maxret_1m | 0.84 | vol_1m (+0.87) |
| minret_1m | 0.80 | vol_1m (−0.83) |
| beta_12m | 0.39 | |
| age | 0.20 | |

The first three are redundant on economic grounds. beta_12m and age carry their own information.

**On the corrected panel**, dropping all five gave unit 1.21 ± 0.08 and account 0.73 ± 0.07 (paired t −0.06).
**Pre-registered on the final panel** (section 10), the eleven-feature candidate scored 0.50 against 0.62 for all 16.

The removal is noise-level either way. It is safe on grounds of simplicity, not of performance.

## 7. Portfolio construction

Reproduce with `research/p3_*.py`.

### 7.1 Short only, then hedged

Net Sharpe; 10% volatility target applied to each book.

| Book | Unit net (base / harsh) | Account Sharpe | Final | Drawdown | Account 2021–26 |
|---|---|---|---|---|---|
| Short only | 0.26 / 0.10 | 0.21 | $0.66m | −31% | |
| Short + long leg (baseline) | 0.81 / 0.61 | 0.68 | $1.61m | −17% | 0.68 |
| Short + volume-weighted universe, 1× | 0.72 / 0.50 | 0.66 | $1.49m | −19% | |
| Short + PGJ, 1× / beta-matched | 0.51 / 0.50 | 0.42 / 0.40 | $1.01m | −31% | 0.70 |
| Short + KWEB (from Aug 2013) | 0.45 | 0.39 | $0.90m | −31% | |
| Short + FXI | 0.39 | 0.30 | $0.80m | −31% | 0.84 |
| Short + MCHI | 0.41 | 0.28 | $0.75m | −31% | 0.77 |
| Short + equal-weighted universe | 0.45 | 0.37 | $0.91m | −25% | |

- **Beta-matching never helps.** The trailing betas average 0.84–1.18 and add noise.
- **Short only:** the shorts earn 21% a year gross but swing 46% a year. 2020 was −69% on the short leg alone.

**Attribution against the volume-weighted universe**, unit book, %/yr:

| Component | Return |
|---|---|
| Short-leg alpha | +32 (beta 0.86) |
| Long-leg alpha | +11 (beta 0.73) |
| Leftover market exposure | −1.7 |
| Costs and borrow | −11.7 |
| **Net** | **29.8** |

An ETF hedge gives up the long-leg alpha and adds mismatch: −5.6% a year with PGJ, −8.1% with FXI.

**Long leg minus PGJ, net Sharpe:**

| | Full period | 2010–20 | 2021–26 |
|---|---|---|---|
| Corrected | 0.56 | 0.90 | 0.02 |
| Shipped | 0.39 | 0.63 | −0.05 |

### 7.2 Why the hedge broke after 2021

The hypothesis: micro-IPOs turned the short leg into tiny, low-beta, idiosyncratic names while the long leg became a copy of PGJ. Yearly averages, 2010–20 → 2021–26:

| | Corrected | Shipped |
|---|---|---|
| Shorts listed < 24 months | 48% → 22% | 44% → 33% |
| Short median price | $15.1 → $9.0 | $16.8 → $6.6 |
| Short beta to PGJ | 1.12 → 1.04 | 1.09 → 0.92 |
| Short idiosyncratic share of variance | 0.51 → 0.49 | 0.55 → 0.60 |
| Long leg correlation with PGJ | 0.80 → 0.79 | 0.78 → 0.83 |
| Long − PGJ, %/yr | +32.2 → +5.5 | +20.0 → +3.3 |

**The hypothesis is mostly rejected.**
- **Shipped data:** weak support. The shorts got lower-beta and more idiosyncratic.
- **Corrected data:** betas and correlations are flat. The micro-IPOs mostly drop out under median volume.
- **What actually changed:** the long leg's stock-picking alpha decayed. It was concentrated in a few years: long minus PGJ was +169% in 2020, +70% in 2017 and +50% in 2013.

### 7.3 Borrow-aware ranking and risk-weighted shorts

| Variant | Unit net | Paired t | Account Sharpe | Worst month | Worst single-name loss |
|---|---|---|---|---|---|
| Baseline (equal weight, forecast rank) | 0.81 | — | 0.68 | −12.2% | −6.4% |
| Borrow-aware, base schedule | 0.65 | −1.33 | 0.58 | −10.4% | |
| Borrow-aware, harsh schedule | 0.59 | −1.49 | 0.54 | | |
| Inverse-volatility shorts (cap 2× equal weight, 15%) | 0.65 | −2.80 | 0.58 | −8.2% | −4.6% |
| Combinations with hedges | 0.08–0.55 | ≤ −1.9 | 0.12–0.53 | | |

Both trims cut the alpha more than they cut costs or tails, because the alpha lives in the expensive, volatile shorts.

With real fees charged, borrow-aware ranking cuts borrow from 19% to 4% a year but leaves the account at Sharpe 0.21, against 0.23 without it (section 8.3).

## 8. Costs and borrow

Reproduce with:
- `research/costs_check.py`
- `research/p4_spreads.py`
- `research/p4_borrow_*.py`
- `research/p4_realistic.py`
- `research/realistic.py`

### 8.1 Trading costs

High-low spread estimates on the names actually held, as half-spread plus commission:

| Leg | Median | Notes |
|---|---|---|
| Long | 23–47 bp | large names (JD, NTES) cost 3–20 bp |
| Short | 31–72 bp | 66–110 bp in 2021–26, over 100 bp below $2 |

- **A tick floor exists.** A one-cent tick on September 2026's shorts (median price $1.95) is already 45–100 bp.
- **Calibration failed.** Live bid/ask quotes couldn't calibrate the estimate: they were taken with the market closed and showed 28–40% spreads. So costs are reported as a range: 0.5× and 1× the Abdi-Ranaldo estimate, plus the tick floor, $0.0035 a share and square-root impact.
- **Capacity:**
  - At $500k a short position is about 0.18% of the name's daily volume.
  - A position reaches 1% of daily volume at $5.4m for the median short, and $1.3m for the 90th percentile.
  - All-in cost per side rises from about 99 bp at $0.5m to about 141 bp at $10m.

### 8.2 Borrow

**Sources:**
- **iBorrowDesk** (IBKR fees and availability): the free API returns 12 months.
- **Nasdaq Reg SHO threshold lists:** 4,167 of 4,381 days, 2010–26.
- **SEC fails-to-deliver:** all files from 2009.
- **FINRA short interest:** from December 2017.
- **NYSE threshold lists** were rate-limited throughout and aren't used.

**Observed IBKR fee on held shorts**, October 2025 to October 2026, n = 150:

| | Value |
|---|---|
| Median / mean / 90th percentile | 67% / 157% / 460% a year |
| Above the harsh schedule | 63% |
| Above 100% a year | 43% |
| Held longs, for contrast | median 0.4% |

- **Volume tiers don't predict cost.** Spike-volume micro-caps in the > $25m tier cost 500–800%; listing age and the size of the collapse predict cost instead.
- **The lender takes the profit.** In the six months where every short has an observed fee (March–August 2026), the gross short leg made +17.9% a month and observed fees left −1.5%.

**Fee model.**
- **Fit:** log fee on volume, listing age, 12-month return, price, idiosyncratic volatility, threshold-list share and fails-to-deliver. R² 0.61, cross-validated by ticker 0.59. It is frozen and capped at 511%.
- **Imputed history:** about 6–19% a year for 2010–19, 25–36% for 2020–24, and 76% for 2025. The 2010–26 average on the shipped book is 37%, against 6.5% assumed. Pre-2025 values are model output, so treat them as indicative.

**Availability rule**, stated before looking at P&L. A short is unavailable if any of:
- IBKR's available shares are below the position, or the observed fee is above 100%;
- the name is on a threshold list for at least half the month;
- the imputed fee is above 100%.

Unavailable names carry:
- 50% of the shipped short leg's gross P&L;
- 40% of the corrected one's;
- 88% in each of 2025 and 2026.

### 8.3 The realistic account

| Fully corrected book | Final | CAGR | Sharpe | 2021–26 | Borrow rate |
|---|---|---|---|---|---|
| As built: 20 bp, assumed borrow | $1.34m | 6.1% | 0.62 | 0.79 | 7.0% |
| Per-name costs at 0.5× / 1× spread, assumed borrow | $0.99m / $0.74m | 4.2% / 2.4% | 0.44 / 0.28 | 0.55 / 0.36 | 7.0% |
| 20 bp, IBKR fee | $0.91m | 3.7% | 0.40 | 0.36 | 19.4% |
| 20 bp, IBKR fee, skip unavailable | $0.92m | 3.7% | 0.40 | 0.24 | 8.7% |
| **0.5× spread, IBKR fee** | **$0.68m** | **1.8%** | **0.23** | 0.13 | 19.4% |
| 0.5× spread, IBKR fee, skip unavailable | $0.69m | 2.0% | 0.24 | 0.03 | 8.7% |
| 0.5× spread, IBKR fee, borrow-aware ranking | $0.65m | 1.6% | 0.21 | −0.11 | 3.7% |
| 1× spread, IBKR fee, skip unavailable | $0.52m | 0.2% | 0.07 | −0.15 | 8.7% |

On the shipped book the same grid gives Sharpe 0.17–0.19 at 0.5× spread with IBKR fees, and −0.02 at 1× with names skipped. A cash account earning the bill rate less 0.5% would have ended at $0.61m ($0.63m at the full bill rate).

**Cash interest** is a fair assumption: about $180–210k, 98% of it in 2022–26. With no short rebate the shipped account still ends at $1.77m.

### 8.4 Do the costs help?

They hurt directly. On the corrected book, trading costs ($180k as built, $330–450k realistic) plus borrow ($240k as built, $450–530k with fees) absorb most of the gross trading profit.

The argument that frictions protect the edge is consistent with what's measured, but they protect it from you as well as from competitors. The edge sits in names whose lenders charge roughly what the shorts earn.

## 9. Sizing

Reproduce with `research/p6_sensitivity.py` (corrected book) and `research/sensitivity.py` (shipped).

**The rule.**
- **Position size:** dollars a side = min(80%, 10% ÷ σ) × equity, where σ = 0.7 × the trailing 60-day volatility of the $1-a-side book + 0.3 × its long-run average.
- **In practice:** the unit book swings 33–37% a year, so the account holds about 26% a side and keeps the rest in bills.

**Fully corrected book:**

| Sizing | CAGR | Sharpe | Drawdown | Peak margin |
|---|---|---|---|---|
| 5% target | 3.7% | 0.71 | −7.6% | 22% |
| **10% target (headline)** | **6.1%** | **0.62** | **−14.7%** | **46%** |
| 15% / 20% target | 8.4% / 10.4% | 0.59 / 0.58 | −21.7% / −28.5% | 73% / 106% |
| Fixed 15% / 25% / 35% / 50% a side | 4.1 / 6.1 / 7.9 / 10.5% | 0.72 / 0.66 / 0.63 / 0.61 | −10.7 / −17.5 / −23.8 / −32.1% | |
| 250-day volatility, no blend | 6.3% | 0.65 | −16.2% | |

**Costs and borrow sensitivity:**

| | Sharpe |
|---|---|
| 5 / 10 / 20 / 40 / 60 / 100 bp per side | 0.72 / 0.68 / 0.62 / 0.49 / 0.36 / 0.10 |
| Borrow: zero / 3% / base / harsh / base × 3 | 0.79 / 0.72 / 0.62 / 0.43 / 0.27 |
| Flat borrow 15% / 25% / 40% | 0.42 / 0.17 / −0.21 |

**Conclusion.** Sizing is a volume knob: Sharpe falls slightly as exposure rises, from cost drag. Costs and borrow decide whether the strategy works at all.

**Daily vs monthly Sharpe.** The shipped account's daily Sharpe (0.75) and monthly Sharpe (1.01) disagree, because daily returns reverse within the month. The corrected book's figures are closer (0.62 vs 0.69). The daily figure is the more honest one.

## 10. Portability: other markets

Reproduce with `research/p5_*.py`.

Identical frozen pipeline and corrections throughout; monthly Sharpe.

| Universe | Rank IC (t) | Rule long-short gross / net | Short leg alone | Beta | 2021–26 | Bottom quintile − EW, %/yr |
|---|---|---|---|---|---|---|
| China, corrected | 0.089 (7.6) | 1.22 / 0.87 | 0.51 | −0.08 | 0.78 | −18 |
| Greater China (HK-based names added) | 0.090 (7.7) | 1.16 / 0.84 | 0.53 | −0.11 | 1.06 | −14 |
| US microcaps (800 sampled) | 0.055 (6.5) | 0.64 / 0.20 | 0.39 | −0.51 | 1.68 | −13 |
| HKEX small caps (1,000 sampled) | 0.050 (5.7) | 0.54 / −0.07 | 0.00 | −0.39 | 0.52 | −7 |
| Non-China Asian micro-listings (≈3 years) | 0.128 (4.7) | too few names for the rule | | | | −41 to −53 |

**Hong Kong natural experiment** (Miller: overpricing persists where shorting is impossible). Excess return against the equal-weight average:

| | Shortable | Not shortable |
|---|---|---|
| Bottom quintile | +0.1% to +1.8% | −22% to −26% |
| Top − bottom quintile | t 2.7 | t 2.4–3.4 |

The pattern holds within size terciles. The confound: the shortable flag is today's list, so it partly selects past winners.

**Reading.**
- **A general effect:** junk small caps lag everywhere.
- **Only China pays:** collected profitably only where the junk is shortable but still under-arbitraged.
- **Elsewhere:** the long-short is mostly short beta.
- **What it supports:** the short-constraint (toxicity-premium) story more than a pure capacity story.
- **Limits:** every new universe is current listings only. The non-China Asian sample is about three years.

## 11. The pre-specified candidate and final checks

Reproduce with:
- `python research/p6_candidate.py drop5|all`
- `python research/p6_deflated.py`

The candidate was written down before it was run (`research/out/p6_candidate_spec.md`): all section 2 fixes; P&L removing only verified errors; the forest with eleven features (the redundant five removed); zero_1m kept; the shipped freeze (section 5.2's rule kept it); the shipped construction.

| | Shipped | Fully corrected, 16 features | Candidate, 11 features |
|---|---|---|---|
| Final equity | $1.81m | $1.34m | $1.08m |
| Sharpe, daily / monthly | 0.75 / 1.01 | 0.62 / 0.69 | 0.50 / 0.56 |
| Worst drawdown | −15.0% | −14.7% | −20.9% |
| Bootstrap 90% interval, monthly Sharpe | 0.62–1.40 | 0.36–1.03 | 0.20–0.95 |
| Random sorts reaching it (of 500) | 0 (max 0.11) | 0 (max 0.15) | 0 (max 0.18) |
| Deflated Sharpe at 13 / 70 / 322 trials | 0.83 / 0.33 / 0.07 | 0.36 / 0.05 / 0.005 | 0.19 / 0.02 / 0.001 |
| Five-year fresh account | $852k (1.00) | $776k (0.83) | $720k (0.70) |
| Rolling five-year Sharpe, latest / median | 1.46 / 0.83 | 0.93 / 0.65 | 0.83 / 0.49 |

**The candidate lost to the 16-feature baseline**, so the eleven-feature version is not recommended. The fully corrected 16-feature book is the reference.

**What these numbers establish.**
- **Better than random:** the ranking clearly beats random sorts.
- **Not skill after the trial count:** once the number of trials is counted honestly, the deflated Sharpe is far below any threshold for skill.
- **Not worth trading at retail:** under realistic frictions (section 8.3) it is close to cash.

## 12. What is still open

- **Historical borrow.** iBorrowDesk's free API covers 12 months, and the Wayback Machine had 2 snapshots. A paid history (S3, Markit, or IBKR's own history through an account) would settle the 2010–24 borrow question.
- **Delisted prices.** Needed for the 266 missing issuers. No free source has them: yfinance returns other companies on reused tickers, Nasdaq's API covers only listed names, and Stooq blocks scripts.
- **Intraday quotes,** to calibrate spreads. The high-low estimate is the main uncertainty in the cost range.
- **NYSE threshold lists** (blocked by Cloudflare throughout).
- **A point-in-time HKEX shortable list,** to remove the confound in the Hong Kong test.
- **Kaggle archive names** use adjusted prices: their split history isn't available, so the price-as-traded test can't be applied to them.

## 13. Trials and caveats

**Trial count.**
- `research/out/trial_log.csv` has 308 logged rows: 252 distinct variants and 24 diagnostics (the rest are repeats and bug reruns).
- The deflated Sharpe uses 322 = the repository's 70 + 252.
- That is a lower bound. The first-round looks of 18 October (models on three panels, the shipped-panel ablation, sizing grids) weren't logged.

**Caveats.**
- **Every comparison uses the same 17 years.** The candidate and the selection rules were pre-specified, but the menu they chose from was not.
- **Noise:** seed noise on the forest is ±0.08–0.12 in Sharpe, and column order moves it by about 0.06. Sharpe's standard error over 188 months is about 0.25.
- **Pre-2025 borrow fees** are imputed from a model fitted to 2025–26. IBKR retail fees are a lower bound for other brokers.
- **Spreads** are high-low estimates, not quotes.
- **Survivorship** affects every universe, most of all 2011–16 here.
- **2026 is a partial year** (to August).

## Appendix: files

All local, none committed.

| What | Where |
|---|---|
| Plan | `research/PLAN.md` |
| Harnesses | `research/account.py` (backtest.py's simulator), `research/realistic.py` (per-name costs, borrow, availability), `research/metrics.py` (standard report), `research/corrected.py` (corrected panel), `research/trials.py` (trial log) |
| First round (18 Oct) | `research/legs.py`, `models.py`, `floors.py`, `clean_events.py`, `lookahead.py`, `panel_variants.py`, `rawpath.py`, `sensitivity.py`, `features_ablation.py`, `costs_check.py`; outputs `legs_*`, `predictions_alt.csv`, `ablation_*`, `sens_*`, `costs_*` |
| Section 2 | `research/p0_splits.py`, `p0_prices.py`, `p0_baseline.py`, `p0_accounts.py`, `p0_reuse.py`; `out/p0_pred_*.csv`, `p0_accounts.csv`, `p0_reuse_compare.csv`, `p0_split_audit.csv`; data `research/data/splits_yf.csv`, `real_prices.csv.gz` |
| Survivorship | `research/p07_*.py`; `out/p07_delisted_china_hk.csv`, `p07_by_year.csv`, `p07_pool_rates.csv` |
| Section 3 | `research/p0_audit.py`; `out/p0_audit_events.csv` (every removal), `p0_audit_asymmetry.csv`, `p0_restored_accounts.csv`; Nasdaq data `research/data/nasdaq/` |
| Section 4 | `research/p0_stress.py`; `out/p0_stress_{corrected,shipped}.csv` |
| Section 5 | `research/p1_*.py`; `out/p1_zero_ic.csv`, `p1_recency.csv`, `p1_selection_choices.csv`, `p1_models*.csv`, `p1_seeds.csv` |
| Section 6 | `research/p2_*.py`; `out/p2_buildup.csv`, `p2_pairs.csv`, `p2_confirm*.csv`, `p2_redundancy*.csv` |
| Section 7 | `research/p3_*.py`; `out/p3_*.csv`; ETF data `research/data/p3/` |
| Section 8 | `research/p4_*.py`; `out/p4_borrow_panel_all.csv`, `p4_fee_model_frozen.json`, `p4_realistic_{shipped,corrected,corrected_reuse}.csv`, `p4_borrow_aware_real.csv`, `p4_spreads.csv`; raw `research/data/borrow/` |
| Section 9 | `research/p6_sensitivity.py`; `out/p6_sens_*.csv` (corrected), `out/sens_*.csv` (shipped) |
| Section 10 | `research/p5_*.py`; `out/p5_summary.csv`, `p5_hk_shortable.csv`, `p5_hkd_book.csv`; data `research/data/p5/` |
| Section 11 | `research/p6_candidate.py`, `p6_deflated.py`; `out/p6_candidate_spec.md`, `p6_{all,drop5}_*.csv`, `p6_deflated.csv`, `trial_log.csv` |
