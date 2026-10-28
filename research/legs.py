"""Leg decomposition with the $3m short rule, monthly unit (1x a side) returns from predictions_rf.csv.
Usage: python research/legs.py [predictions.csv] [col]"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "."); from adr_longshort import load_predictions, load_benchmarks, load_rf, holdings
path = sys.argv[1] if len(sys.argv) > 1 else None; col = sys.argv[2] if len(sys.argv) > 2 else "p_RF"
F = load_predictions(path); L = F[F.liquid]; B = load_benchmarks(); _, rfm = load_rf()
BASE = [(25e6, .01), (10e6, .04), (3e6, .12), (0, .35)]; HARSH = [(25e6, .02), (10e6, .08), (3e6, .25), (0, .60)]; C = 0.002
tier = lambda a, s: next(r for lo, r in s if a >= lo)
rows = []; p5 = p1 = set()
for ym, g in L.groupby("ym"):
    h = holdings(g, col)
    if h is None: continue
    lo = g[g.ticker.isin(h["long"])]; sh = g[g.ticker.isin(h["short"])]; s5, s1 = set(h["long"]), set(h["short"])
    pg = B["PGJ"].get(ym + 1, np.nan) - rfm.get(ym + 1, np.nan); kw = B["KWEB"].get(ym + 1, np.nan) - rfm.get(ym + 1, np.nan)
    rows.append(dict(ym=ym, L=lo.exret_next.mean(), S=sh.exret_next.mean(), EW=g.exret_next.mean(), PGJ=pg, KWEB=kw,
                     to5=1 - len(s5 & p5) / len(s5) if p5 else 1, to1=1 - len(s1 & p1) / len(s1) if p1 else 1,
                     bb=np.mean([tier(a, BASE) for a in sh.adv]) / 12, bh=np.mean([tier(a, HARSH) for a in sh.adv]) / 12, nS=len(sh), nL=len(lo)))
    p5, p1 = s5, s1
q = pd.DataFrame(rows).set_index("ym")
def st(x, name):
    x = x.dropna(); sr = lambda y: y.mean() / y.std() * np.sqrt(12); c = (1 + x).cumprod()
    return dict(strategy=name, ret_yr=x.mean() * 1200, vol=x.std() * np.sqrt(12) * 100, sharpe=sr(x), sh_2010_15=sr(x.loc["2010":"2015"]), sh_2016_20=sr(x.loc["2016":"2020"]), sh_2021_26=sr(x.loc["2021":]),
                maxdd=(c / c.cummax() - 1).min() * 100, worst_m=x.min() * 100, beta_pgj=np.cov(x, q.PGJ.reindex(x.index))[0, 1] / q.PGJ.reindex(x.index).var())
cL = C * 2 * q.to5; cS = C * 2 * q.to1; etf = 0.0004   # ETF leg: ~2 bp each way, rebalanced monthly; 1% borrow when shorting the ETF
out = [st(q.L - q.S, "long-short gross"), st(q.L - q.S - cL - cS - q.bb, "long-short net (base borrow)"), st(q.L - q.S - cL - cS - q.bh, "long-short net (harsh borrow)"),
       st(-q.S, "short leg alone gross (naked)"), st(-q.S - cS - q.bb, "short leg alone net (naked)"),
       st(q.PGJ - q.S, "short leg + long PGJ gross"), st(q.PGJ - q.S - cS - q.bb - etf, "short leg + long PGJ net"),
       st(q.KWEB - q.S - cS - q.bb - etf, "short leg + long KWEB net"), st(q.EW - q.S, "short leg + long EW liquid universe gross"),
       st(q.L, "long leg alone gross"), st(q.L - q.PGJ - cL - 0.01 / 12 - etf, "long leg - PGJ net"), st(q.PGJ, "PGJ excess")]
T = pd.DataFrame(out).set_index("strategy"); pd.set_option("display.width", 250); print(T.round(2).to_string())
print(f"\nmonths {len(q)}, avg longs {q.nL.mean():.1f}, avg shorts {q.nS.mean():.1f}; KWEB from {q.KWEB.first_valid_index()}")
print("long-short gross: share of return from short side = %.0f%%" % (100 * -q.S.mean() / (q.L - q.S).mean()))
q.to_csv("research/out/legs_monthly.csv"); T.to_csv("research/out/legs_stats.csv")
