# Phase 6 candidate specification (written before the candidate is run)

Fixed 19 October 2026, after phases 0, 2, 3, 4 and 5 reported and before phase 1 reported. The phase 1 item is decided by the
rule stated here, not by the candidate's own result.

## Data and universe (phase 0, adopted regardless of effect)
- Price ≥ $1 on the price as actually traded (yfinance splits undone); no "adjusted price ever > $5,000" exclusion.
- Every liquidity threshold ($100k fitting universe, $1m long, $3m short) on the month's median daily dollar volume.
- No "≥ 10 trading days next month" filter for scoring; only rows where the data ends because the Kaggle archive ends are excluded.
- Prices before 1 January of (IPO year − 1) dropped (ticker reuse, pre-merger shells).
- P&L and margin: only verified data errors are removed (the phase 0.4 audit); real and unverifiable moves stay.

## Model and features
- Random forest, shipped protocol and grid.
- Features: the 16 minus beta_12m, minret_1m, illiq_1m, maxret_1m, age (11 features). Phase 2.4: removal is neutral within noise
  (paired t −0.06, seed spread ±0.08); adopted for simplicity, not for performance.
- zero_1m kept: phase 2.3 shows dropping it is neutral on Sharpe inside the protocol and costs 2021–26; not adopted.
- Phase 1 refit/model choice: adopted only if phase 1's validation-selected path (chosen inside the protocol) has paired t > −1 against the
  shipped protocol on the corrected baseline and holds in at least two of the three sub-periods; otherwise the shipped protocol stays.

## Portfolio construction (phase 3: no change)
- Long top quintile of ≥ $1m, short bottom quintile of ≥ $3m, equal weight, forecast rank only, monthly; 10% volatility target, 80% cap.

## Reported for the candidate
- The account as built (20 bp, base tiered borrow), and the realistic account (per-name costs at 0.5× and 1× the high-low spread,
  observed-or-imputed IBKR fees, with and without skipping unavailable names).
- Block bootstrap of Sharpe, random-sort null, deflated Sharpe at the full trial count (the repo's 70 plus every variant logged here).
- Five-year fresh account (Sep 2021 – Aug 2026).
