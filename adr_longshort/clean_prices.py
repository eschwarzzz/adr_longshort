"""Shared daily-price cleaning. (1) Drop isolated bad prints: a close more than 3x or less than 1/3 of the previous close that reverts the next day.
(2) Pair rule: a day below -75% followed within 5 trading days by a day above +300% is a print error or an unadjusted split pair; both days' returns are removed.
(3) Any remaining single day above +300% (an unadjusted reverse split) is removed, as before.
The frame is re-indexed first: with duplicate index labels (a concatenation of sources) the label-based assignments would null returns in unrelated rows."""
import numpy as np, pandas as pd
def clean(P, close="close", ticker="ticker", date="date"):
    P = P.sort_values([ticker, date]).reset_index(drop=True)        # a unique index: the .loc assignments below are by label, and concatenated inputs carry duplicate labels
    g = P.groupby(ticker)[close]; r1 = P[close] / g.shift(1); r2 = g.shift(-1) / P[close]
    spike = ((r1 > 3) & (r2 < 1 / 3)) | ((r1 < 1 / 3) & (r2 > 3)); P = P[~spike.fillna(False)].copy()
    P["ret"] = P.groupby(ticker)[close].pct_change(); fwd = pd.concat([P.groupby(ticker)["ret"].shift(-k) for k in range(1, 6)], axis=1).max(axis=1)
    crash_pair = (P["ret"] < -0.75) & (fwd > 3); idx = P.index[crash_pair.fillna(False)]
    P.loc[idx, "ret"] = np.nan
    for k in range(1, 6):
        nxt = P.groupby(ticker)["ret"].shift(-k); hit = crash_pair.fillna(False) & (nxt > 3); P.loc[P.index[hit].map(lambda i: i), "ret"] = np.nan        # the crash day
    rebound = P["ret"] > 3; P.loc[rebound.fillna(False), "ret"] = np.nan
    return P, dict(spikes=int(spike.fillna(False).sum()), crash_pairs=int(crash_pair.fillna(False).sum()), big_jumps=int(rebound.fillna(False).sum()))
