# Data

Snapshot of 9 October 2026. `fetch_data.py` refreshes everything except the Kaggle archive.

| File | Content | Source |
|---|---|---|
| `universe.csv` | 453 US-listed Chinese companies: ticker, name, IPO year, first and last list appearance, source | USCC lists of 2019, 2020 and 2025 (PDFs) and the Nasdaq screener (country = China) |
| `yf_prices.csv.gz` | daily adjusted OHLC and volume for the 302 currently listed names, 1999 to 8 October 2026 | Yahoo Finance via `yfinance` (`auto_adjust=True`). Three tickers now reused by other companies (ACH, ATAI, BITA) are excluded here and taken from the Kaggle archive |
| `kaggle_patch/stocks/*.csv` | daily OHLC, adjusted close and volume for 105 names that delisted or were renamed before April 2020 | Kaggle `jacksoncrow/stock-market-dataset` (CC0), histories to 1 April 2020. Twelve SOE ADRs (PTR, SNP, LFC, SHI, ACH, GSH, HNP, CEA, ZNH, CHL, CHU, CEO) are extended from April 2020 to their NYSE delisting dates with the daily returns of their Hong Kong lines, volume carried forward from the ADR's last 60 days scaled by the HK line's volume |
| `fred_DTB4WK.csv` | 1-month Treasury bill rate, daily, percent per year | FRED series DTB4WK |
| `benchmarks_monthly.csv` | monthly total returns of FXI, KWEB, MCHI, PGJ and SPY | Yahoo Finance, adjusted closes |

46 universe names have no price history in either source (SPACs and names listed only briefly) and do not enter the panel.

Price cleaning (`adr_longshort/clean_prices.py`, applied at load time by the panel builder): isolated prints that revert the next day are dropped, both days of a crash-then-rebound pair are nulled, and any remaining single day above +300% (an unadjusted reverse split) is nulled. Names whose adjusted price ever exceeds $5,000 (serial reverse-splitters) are excluded from the panel.
