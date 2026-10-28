# adr_longshort: analysis plan

**Draft, 19 October 2026.** Covers every test still open from the review. It ends with an inspection-notes
document and a final recommendation document in the euribor_momentum style. Local only; nothing committed.

## Ground rules

1. **Fix the baseline first, then test changes on it.** The look-ahead and liquidity fixes (phase 0) change which
   names are tradeable. Every later comparison runs against the corrected baseline. The shipped numbers stay
   alongside, so each change is credited separately.
2. **Selection happens inside the protocol.** A choice of model, feature set or refit scheme is made each December
   on the validation years, the same way tree depth is chosen today. Test-period tables describe; they don't choose.
3. **Log every trial.** Each variant run gets a line in `research/out/trial_log.csv`. The final deflated Sharpe
   uses the true count: the repo's 70, plus ours.
4. **Judge on the whole strategy, not Sharpe alone.** Each variant reports:
   - CAGR, volatility, daily and monthly Sharpe, worst drawdown, worst month and worst day
   - Beta to PGJ, KWEB and the volume-weighted universe
   - Return of each leg separately
   - Turnover, costs, borrow and margin peak
   - Sub-periods 2010–15, 2016–20 and 2021–26
   - Paired t-stat of the monthly difference against the baseline, and spread across random seeds for anything fitted
5. **Adoption rule, fixed in advance.** Adopt a change only if all three hold:
   - It was chosen inside the protocol, or is pre-specified on economic grounds.
   - It doesn't worsen net Sharpe or drawdown by more than noise: paired t above −1, and seed spread respected.
   - It holds in at least two of the three sub-periods.

   Data-correctness fixes (phase 0) are adopted regardless of their effect on performance.

## Already done (reused, not rerun)

| Item | Result | Where |
|---|---|---|
| Reproducibility | Tests pass; refit matches the shipped forecasts to 1e-16 | — |
| Leg decomposition, ETF hedge (monthly unit book) | Short leg is 68% of the gross spread; long−PGJ net 0.34, −0.10 in 2021–26 | `out/legs_*` |
| Huber / OLS / simple rules (IC and unit book) | Huber IC 0.115 but long-short 0.68; OLS 1.31; RF 1.37 | `out/predictions_alt.csv` |
| Short-floor sweep by model | Every model peaks at $3m | `floors.py` |
| Refit on train+validation | No gain (1.26 vs 1.37) | `p_RFTV` |
| Feature ablation (drop-one, families, greedy, permutation, seeds) | zero_1m hurts (1.69 without it, seeds 1.60 ± .07); 5 features look redundant | `out/ablation_*` |
| Sizing, volatility estimator, cost and borrow grids | Sizing changes return, not Sharpe; costs and borrow decide the strategy | `out/sens_*` |
| Spreads, capacity, borrow profile, cash interest | Short-leg spreads 40–80 bp; 79% of short P&L from hard-to-borrow names; borrow break-even 41% | `out/costs_*` |
| Cleaning events in held positions (20 days) | All reverted within a day or two; raw P&L 0.72 vs 0.75 | `out/clean_events_held.csv` |
| Look-ahead scan | 55 names dropped by the $5,000 rule; 27 rows by the next-month filter | `lookahead.py` |
| Spike-inflated volume | 13% of short name-months (21% since 2021) pass $3m only on mean volume | — |

## Phase 0: corrected baseline (data and universe)

**0.1 Split history and real traded prices.**
- Pull split events from yfinance for the 302 listed names.
- For the archive names, infer splits from the price-and-volume signature: price jumps by k and stays, volume
  drops by k. GMM is the template.
- Rebuild each name's price as it actually traded.
- Replace the $5,000 rule with a point-in-time test: tradeable only if the price as traded was ≥ $1 that month.
  This brings back the 55 serial reverse-splitters in the months they were genuinely tradeable.

**0.2 Halts and delistings.**
- Drop the "≥ 10 trading days next month" filter.
- A name that stops trading while held stays in the book at its last price. Borrow keeps accruing on shorts.
- Exit rule, fixed in advance: a long is marked at −100% on delisting. A short is closed at the last price plus a
  30% buy-in haircut if trading resumes, and at 0 otherwise.
- Report how often this binds.

**0.3 Liquidity measured on typical days.**
- The $1m and $3m floors use the month's median daily dollar volume instead of the mean (a trimmed mean as a check).
- Pre-specified: the economics, not the backtest, choose the median.

**0.4 Cleaning, audited.** One log row per removal, with these fields:
- ticker, date, raw return, which filter fired
- whether the name was long, short or not held, and the position size
- the second archive's return where the two overlap
- split record, price after 10 and 20 days, volume ratio

Classify each removal as a verified error (split, revert print, archive disagreement), a real move, or unknown.

| Use of the data | Rule |
|---|---|
| Training | Removing verified errors is fine; a training filter may be picked on validation |
| P&L and margin | Only verified errors are removed; real and unknown moves are restored |

Then:
- **Asymmetry table:** removals by leg and direction (up vs. down, short vs. long).
- **Restored account:** worst month, worst day, and days with a maintenance-margin call.

**0.5 Stress tests.**
- Each month, shock the largest short +200% on its worst day. Report drawdown, margin and forced-cover loss.
- Repeat at the 5-name minimum.
- **AMTD Digital (HKD), mid-2022:** it is *not in the universe* (Hong Kong-based, missed by the USCC lists). Add it
  and other HK-based names. Would it have been shorted? What would the spike have done?

**0.6 Corrected baseline.** Rebuild the panel, refit, backtest and run the Monte Carlo on phase-0 data. Report
shipped vs. corrected side by side, with each fix's effect shown on its own. **Everything after this runs on the
corrected baseline.**

## Phase 1: model and protocol (on the corrected baseline)

**1.1 Where zero_1m matters.** Rank IC and quintile spreads of zero_1m, and of every other feature, computed
within the ≥ $100k, ≥ $1m and ≥ $3m sets separately. Its importance score describes the fitting universe, not the
traded one.

**1.2 Refit recency, chosen inside the protocol.** Candidates:
- the shipped freeze (training ends Y−3)
- a one-year validation window (training ends Y−2)
- a rolling 6- or 8-year training window
- recency-weighted fitting
- quarterly refits

Validation picks among them each December. Report the selected-variant path against the shipped freeze.

**1.3 Forest vs. Huber vs. OLS vs. a 50/50 blend.**
- The full $500k account for each, with the same sizing, costs and floor. This is the row missing from the doc.
- Score each where the money is: top-minus-bottom spread within the $1m long set and within the $3m short set,
  separately.
- Paired t-stat of forest minus Huber, and forest minus OLS, over about 200 months.
- Then make model choice part of the protocol: each December, validation picks forest, Huber, OLS or the blend.

## Phase 2: features (on the corrected baseline, with settings re-chosen each year)

**2.1 Build-up path.**
- Start with high52 plus the four momentum features.
- Add zero_1m.
- Then add each remaining feature one at a time, in a fixed order.
- Finish with all 16.

Full account metrics at every step.

**2.2 Pairs.**
- The 8 features that carry the result (high52, ivol_12m, mom_6_1, dolvol_trend, mom_12_1, vol_1m, dolvol_1m,
  zero_1m): all 28 pairs as two-feature forests, plus each pair added to the base set.
- Look for features that only help together.

**2.3 Confirm dropping zero_1m inside the protocol.** Validation decides each December whether zero_1m stays.
Report full account metrics and beta, not just Sharpe.

**2.4 The redundant five** (beta_12m, minret_1m, illiq_1m, maxret_1m, age). Drop them together and one at a time,
with account metrics. Adopt only if performance is unchanged within noise.

## Phase 3: portfolio construction

**3.1 Attribution: short only, then hedged.** All on the full period 2010–26, monthly rebalance, net of costs:

| Book | Hedge |
|---|---|
| Short leg | none |
| Short leg | PGJ at hedge ratio 1, and beta-matched (trailing 12-month daily beta, point-in-time) |
| Short leg | KWEB (from Aug 2013), FXI, MCHI, beta-matched |
| Short leg | volume-weighted and equal-weighted universe baskets (closest size match to the shorts) |
| Short leg | the current long leg |

- Report net Sharpe, CAGR, drawdown, beta, hedge cost and tracking error, by year.
- Split total P&L into short-leg alpha, hedge or market, and long-leg alpha.
- Run as the $500k account as well as the unit book.

**3.2 Why the hedge broke after 2021.**
- **Hypothesis:** the micro-IPO wave changed what sits in the short leg. Shorts became tiny, idiosyncratic names
  with low beta to large-cap ETFs. Meanwhile the long top quintile (JD, NTES, BEKE) became a near-copy of PGJ.
- **Tests, by year:**
  - share of shorts listed under 24 months
  - median market cap and volume of the shorts
  - short leg's rolling beta to PGJ, and the idiosyncratic share of its variance
  - correlation of the long leg with PGJ
  - long-leg return minus the volume-weighted universe

**3.3 Borrow-aware short ranking.**
- Rank shorts by forecast minus expected monthly borrow (rate ÷ 12).
- Use the base and harsh schedules, then the phase-4 borrow proxy.
- Names flagged unavailable are skipped, and the next-ranked name is taken.

**3.4 Risk-weighted shorts.**
- Inverse-volatility weights (60-day volatility, point-in-time) with a hard cap per name of 2× the equal weight,
  and an absolute cap of 15% of the leg.
- Compare worst day, worst month, largest single-name loss and the 0.5 shock test, as well as Sharpe.

## Phase 4: costs and borrow, confirmed

**4.1 Per-name trading costs in the account.**
- Replace the flat 20 bp with half-spread (Abdi-Ranaldo, floored at one tick) plus commission.
- Square-root impact at the account size.

**4.2 Borrow evidence, from free sources** (check availability and terms first):
- **iBorrowDesk:** IBKR fee and availability history for held shorts, about 2017 onward.
- **Reg SHO threshold lists** (Nasdaq and NYSE daily archives): a hard-to-borrow flag.
- **SEC fails-to-deliver files:** a stress flag.
- **FINRA short interest:** crowding.

Build a per-name borrow estimate and availability flag; calibrate the tiered schedule against it.

**4.3 Rerun the account** with the realistic cost model, the borrow estimate, and the skip-if-unavailable rule.
This is the "realistic" headline.

## Phase 5: portability (free data, hypothesis-level)

The question: is this a general "toxic small-cap" premium (portable, but a risk premium), or something specific to
China (which regulators can close)?

| Test | Universe and source | Known limits |
|---|---|---|
| 5.1 Non-China foreign micro-IPOs | Nasdaq listings of Singapore, Malaysia, HK and Cayman-incorporated firms, about 2019–26; yfinance | Short sample, survivorship |
| 5.2 US microcaps | Same volume band; the Kaggle archive (US stocks to April 2020) plus yfinance for survivors | Survivorship after 2020 |
| 5.3 Hong Kong small caps | yfinance `.HK` prices; HKEX's list of designated securities as the shortable flag | A natural experiment on short constraints |

- Run the identical pipeline: same 16 features, same protocol, frozen settings.
- No tuning on these universes.
- Report what the data can and can't support.

## Phase 6: candidate specification and final checks

**6.1** Combine the adopted changes into one candidate spec, written down before it is run.
**6.2** Run the account, Monte Carlo, random-sort null and the deflated Sharpe at the full trial count.
**6.3** Five-year fresh account, and paper-trade the months after August 2026 as a hold-out.

## Deliverables (mirroring euribor_momentum)

1. **`reports/INSPECTION.md`:** numbered, versioned inspection notes.
   - Opens with a summary list.
   - One section per phase, each giving the command that reproduces it.
   - Ends with trials and caveats, and an appendix of files.
2. **Final recommendation document**, as `REVIEW.docx` and `REVIEW.pdf` (about 10 pages):
   - Summary with the recommendation
   - 1. What holds up
   - 2. Changes we recommend
   - 3. Smaller suggestions
   - 4. Questions for the author
   - 5. Concerns to keep in view
   - Appendices with the evidence

## Order and rough effort

| Phase | Depends on | Effort |
|---|---|---|
| 0 Corrected baseline | — | Largest; split history and audit are manual-heavy |
| 1 Model and protocol | 0 | Moderate (refits) |
| 2 Features | 0, and 1.2 for the protocol | Moderate |
| 3 Construction | 0 | Light to moderate |
| 4 Costs and borrow | 0; feeds 3.3 | Data gathering is the risk |
| 5 Portability | Can run in parallel after 0 | Moderate; data-limited |
| 6 Candidate and report | All | Light |
