"""Phase 0.7d: how often surviving names of each listing cohort entered the liquid pools (corrected panel, median ADV), to scale the missing names."""
import pandas as pd, numpy as np
M = pd.read_csv("research/out/p0_panel_corrected.csv", usecols=["ticker", "ym", "adv", "tradeable", "score_ok"]); M["ym"] = pd.PeriodIndex(M.ym, freq="M")
U = pd.read_csv("data/universe.csv"); ipo = pd.to_numeric(U.set_index("ticker").ipo_year, errors="coerce"); M["ipo"] = M.ticker.map(ipo)
S = M[M.score_ok]
def coh(lo, hi, win):
    names = ipo[(ipo >= lo) & (ipo <= hi)].index; s = S[S.ticker.isin(names) & (S.ym.dt.year >= lo) & (S.ym.dt.year <= hi + win)]
    g = s.groupby("ticker").adv; n = len(names); seen = s.ticker.nunique()
    return dict(cohort=f"IPO {lo}-{hi}", names_in_universe=n, names_in_panel=seen, ever_ge_1m=(g.max() >= 1e6).sum(), ever_ge_3m=(g.max() >= 3e6).sum(),
                months_ge_3m_per_name=float((s.adv >= 3e6).groupby(s.ticker).sum().mean()) if seen else np.nan)
T = pd.DataFrame([coh(2004, 2010, 3), coh(2011, 2015, 3), coh(2016, 2019, 3), coh(2020, 2022, 3), coh(2023, 2026, 3)]); T.to_csv("research/out/p07_pool_rates.csv", index=False); print(T.to_string(index=False))
H = S.groupby("ym").apply(lambda g: (g.adv >= 3e6).sum(), include_groups=False); print("short pool size (>= $3m median), mean by year:", H.groupby(H.index.year).mean().round(0).to_dict())
