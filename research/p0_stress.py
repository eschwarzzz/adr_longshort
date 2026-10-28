"""Phase 0.5: intraday and shock stress on the short book.
(a) Historical intraday: each held short's daily HIGH vs the previous close; account equity if every short hit its high at once (conservative),
    Reg T maintenance (25% long, 30% short) at that point; would-be calls.
(b) Shock: each month, the largest short goes +200% on one day mid-month (temporary or persistent): account loss, maintenance; also at the 5-name minimum.
(c) What drove the worst account day (2011-07-18)."""
import sys, glob, os, numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "research")
from adr_longshort import load_predictions, holdings
from trials import log
Y = pd.read_csv("data/yf_prices.csv.gz", parse_dates=["date"]).rename(columns=str.lower)[["ticker", "date", "high", "close"]]
K = []
for f in sorted(glob.glob("data/kaggle_patch/stocks/*.csv")):
    t = os.path.basename(f)[:-4]; k = pd.read_csv(f, parse_dates=["Date"]); K.append(pd.DataFrame({"ticker": t, "date": k.Date, "high": k.High * k["Adj Close"] / k.Close, "close": k["Adj Close"]}))
K = pd.concat(K); K = K[~K.ticker.isin(Y.ticker.unique())]; P = pd.concat([Y, K]).sort_values(["ticker", "date"])
P["prev"] = P.groupby("ticker").close.shift(1); P["hi_ret"] = P.high / P.prev - 1; P["ym"] = P.date.dt.to_period("M")
ev = pd.read_csv("research/out/p0_audit_events.csv", parse_dates=["date"]); bad = set(zip(ev[ev.verdict == "verified error"].ticker, ev[ev.verdict == "verified error"].date))
P = P[[(t, d) not in bad for t, d in zip(P.ticker, P.date)]]          # verified bad prints out; real spikes stay
out = {}
for book, pred, mon_f in [("corrected", "research/out/p0_pred_corrected.csv", "research/out/p0_monthly_corrected.csv"), ("shipped", "results/predictions_rf.csv", "results/bt500k_rf_monthly.csv")]:
    F = load_predictions(pred); mon = pd.read_csv(mon_f, parse_dates=["date"]); mon["hm"] = mon.date.dt.to_period("M") + 1; mm = mon.set_index("hm"); rows = []
    for ym, g in F[F.liquid].groupby("ym"):
        h = holdings(g); hm = ym + 1
        if not h or hm not in mm.index or mm.exposure.get(hm, 0) == 0: continue
        E = mm.equity_start[hm]; side = mm.dollars_side[hm]; ns = len(h["short"]); w = side / ns
        x = P[(P.ym == hm) & P.ticker.isin(h["short"])]
        worst = x.groupby("ticker").hi_ret.max()                                   # worst intraday rise of each short in the month
        day = x.groupby("date").apply(lambda d: (w * d.hi_ret.clip(lower=0)).sum(), include_groups=False)   # all shorts at their highs that day
        loss_all = day.max() if len(day) else 0.0
        maint = (0.25 * side + 0.30 * (side + loss_all)) / max(E - loss_all, 1)
        shock_loss = 2 * w; maint_shock = (0.25 * side + 0.30 * (side + 2 * w)) / (E - 2 * w)
        w5 = side / 5; shock5 = 2 * w5; maint5 = (0.25 * side + 0.30 * (side + 2 * w5)) / (E - 2 * w5)
        rows.append(dict(hm=str(hm), equity=E, side_pct=side / E * 100, n_short=ns, worst_single_intraday=worst.max() if len(worst) else np.nan, worst_name=worst.idxmax() if len(worst) else "",
                         intraday_all_highs_loss_pct=loss_all / E * 100, intraday_maint=maint, shock200_loss_pct=shock_loss / E * 100, shock200_maint=maint_shock, shock200_at5_loss_pct=shock5 / E * 100, shock200_at5_maint=maint5))
    T = pd.DataFrame(rows); T.to_csv(f"research/out/p0_stress_{book}.csv", index=False); out[book] = T
    print(f"\n== {book} book: {len(T)} months")
    print(f"worst single-name intraday rise: {T.worst_single_intraday.max()*100:.0f}% ({T.loc[T.worst_single_intraday.idxmax(), 'worst_name']} {T.loc[T.worst_single_intraday.idxmax(), 'hm']}); months with a short up >100% intraday: {(T.worst_single_intraday > 1).sum()}")
    print(f"all shorts at their highs on the worst day: max loss {T.intraday_all_highs_loss_pct.max():.1f}% of equity; peak maintenance {T.intraday_maint.max():.2f} of equity (call if > 1); calls {int((T.intraday_maint > 1).sum())}")
    print(f"+200% shock on one short: loss median {T.shock200_loss_pct.median():.1f}%, max {T.shock200_loss_pct.max():.1f}% of equity; maintenance max {T.shock200_maint.max():.2f}; calls {int((T.shock200_maint > 1).sum())}")
    print(f"  at the 5-name minimum: loss median {T.shock200_at5_loss_pct.median():.1f}%, max {T.shock200_at5_loss_pct.max():.1f}%; maintenance max {T.shock200_at5_maint.max():.2f}")
    print(T.nlargest(5, "worst_single_intraday")[["hm", "worst_name", "worst_single_intraday", "n_short", "side_pct", "intraday_all_highs_loss_pct"]].round(3).to_string(index=False))
    log("P0", f"stress test {book} book", f"worst intraday single {T.worst_single_intraday.max():.2f}, shock200 max loss {T.shock200_loss_pct.max():.1f}%", f"research/out/p0_stress_{book}.csv")
# (c) the worst account day
d = pd.read_csv("research/out/p0_daily_corrected.csv", parse_dates=["date"], index_col="date"); dd = d.ret.idxmin()
F = load_predictions("research/out/p0_pred_corrected.csv"); hm = dd.to_period("M"); g = F[F.liquid & (F.ym == hm - 1)]; h = holdings(g)
R = P[(P.date == dd)].set_index("ticker").assign(r=lambda x: x.close / x.prev - 1).r
print(f"\nworst account day {dd.date()} ({d.ret.min()*100:.1f}%): long-leg mean {R.reindex(h['long']).mean()*100:.1f}%, short-leg mean {R.reindex(h['short']).mean()*100:.1f}%")
print("  shorts that day:", R.reindex(h["short"]).round(3).to_dict())
