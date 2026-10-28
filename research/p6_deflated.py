"""Merge the trial logs; deflated Sharpe (Bailey & Lopez de Prado) at the full trial count, with the repo's dispersion of trial Sharpes."""
import glob, numpy as np, pandas as pd
from scipy.stats import norm, skew, kurtosis
import csv
rows = []
for f in sorted(glob.glob("research/out/trial_log_*.csv")):
    for r in csv.reader(open(f)):
        if not r or r[0] == "trial_id": continue
        if len(r) >= 6: rows.append(dict(trial_id=r[0], phase=r[1], description=r[3], metric_summary=r[4], file=r[5], log=f.split("/")[-1]))
        elif len(r) == 5: rows.append(dict(trial_id=r[0], phase=r[1], description=r[2], metric_summary=r[3], file=r[4], log=f.split("/")[-1]))
L = pd.DataFrame(rows)
L["phase"] = L.phase.astype(str).str.upper().str.replace("-", ""); L.to_csv("research/out/trial_log.csv", index=False)
diag = L.description.astype(str).str.contains("stress|audit|coverage|diagnostic|restored|hedge-break|baseline variant|build", case=False)
distinct = L[~diag].drop_duplicates("description")
print(f"logged rows {len(L)}; diagnostics {int(diag.sum())}; distinct variant trials {len(distinct)}"); print(distinct.groupby("phase").size().to_string())
tr = pd.read_csv("results/research_quintile_returns.csv"); V = tr.assign(ls=tr.q5 - tr.q1).groupby("strategy").ls.apply(lambda s: s.mean() / s.std()).var(); g = 0.5772156649
def dsr(x, n):
    sr0 = np.sqrt(V) * ((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * np.e))); sr = x.mean() / x.std()
    return norm.cdf((sr - sr0) * np.sqrt(len(x) - 1) / np.sqrt(1 - skew(x) * sr + (kurtosis(x, fisher=False) - 1) / 4 * sr ** 2)), sr0 * np.sqrt(12)
rows = []
for lab, f in [("shipped", "results/bt500k_rf_monthly.csv"), ("corrected, 16 features", "research/out/p6_all_monthly.csv"), ("candidate, 11 features", "research/out/p6_drop5_monthly.csv")]:
    x = pd.read_csv(f).ret.dropna()
    for n in (13, 70, 70 + len(distinct)):
        p, sr0 = dsr(x, n); rows.append(dict(account=lab, trials=n, sharpe_monthly=x.mean() / x.std() * np.sqrt(12), expected_max_null=sr0, deflated_prob=p))
T = pd.DataFrame(rows); T.to_csv("research/out/p6_deflated.csv", index=False); print(T.round(3).to_string(index=False))
