"""Held names by month (formation month ym, held in ym+1), $3m floor, from shipped predictions. -> research/out/p4_held.csv"""
import sys, pandas as pd
sys.path.insert(0, "."); from adr_longshort import load_predictions, holdings
F = load_predictions(); L = F[F.liquid]; rows = []
for ym, g in L.groupby("ym"):
    h = holdings(g)
    if not h: continue
    gi = g.set_index("ticker")
    for side in ("long", "short"):
        for t in h[side]: rows.append(dict(ticker=t, ym=str(ym), hold_month=str(ym + 1), side=side, adv=gi.adv[t], exret_next=gi.exret_next[t], n_side=len(h[side])))
pd.DataFrame(rows).to_csv("research/out/p4_held.csv", index=False)
H = pd.DataFrame(rows); print(len(H), "rows;", H[H.side=="short"].ticker.nunique(), "short tickers;", H.ticker.nunique(), "all tickers")
