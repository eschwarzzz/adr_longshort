"""Same holdings, P&L from raw (uncleaned) daily closes: does any intramonth spike breach maintenance margin or change the drawdown?"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "research"); from account import load
for mode in ("repo", "none"):
    B = load(clean_mode=mode); d, m = B.simulate(); s = B.stats(d, m)
    worst_day = d.ret.idxmin()
    print(f"{mode:5s} final ${s['final_equity']:,.0f} sharpe {s['sharpe']:.2f} maxdd {s['maxdd_pct']:.1f}% worst day {d.ret.min()*100:.1f}% on {worst_day.date()} best day {d.ret.max()*100:.1f}% peak maint {s['peak_maint_pct']:.0f}% call days {s['call_days']}")
    print("   5 worst days:", {str(k.date()): round(v * 100, 1) for k, v in d.ret.nsmallest(5).items()})
# what made the -70% month in the uncleaned OLS run
q = pd.read_csv("research/out/predictions_noclean.csv"); q["ym"] = pd.PeriodIndex(q.ym, freq="M")
print(q[q.liquid].nlargest(5, "exret_next")[["ticker", "ym", "adv", "exret_next", "p_OLS", "p_RF"]].to_string(index=False))
