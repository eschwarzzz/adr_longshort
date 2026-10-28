"""Phase 0.4: every cleaning removal, logged and classified; asymmetry by leg; P&L restored for real/unknown moves; margin.
Classification rules (fixed before looking at P&L), first match wins:
  E1 split listed      : a yfinance split within 5 trading days                                   -> verified error
  E2 split unlisted    : >=3x jump that persists 10 days, volume moves the opposite way >= 2x        -> verified error
  E3 mixed scales      : price flips by >=3x and back at least twice within 10 days                 -> verified error
  E4 second source     : Nasdaq historical close disagrees with the Yahoo move by > 2x               -> verified error
  E5 outside the range : close outside the day's own high-low by > 1%                                -> verified error
  R1 real              : second source confirms the move (within 2x) or volume >= 3x the 20-day median with the high confirming
  U  unknown           : anything else (kept in the P&L as real: only verified errors may leave it)
Writes research/out/p0_audit_events.csv, p0_audit_asymmetry.csv, p0_restored_accounts.csv; logs trials."""
import sys, os, json, time, glob, numpy as np, pandas as pd, requests
sys.path.insert(0, "."); sys.path.insert(0, "research")
from adr_longshort import load_predictions, holdings
from account import load_ns
from trials import log
# ---- prices with OHLC (adjusted), both sources, exactly as load_prices combines them
Y = pd.read_csv("data/yf_prices.csv.gz", parse_dates=["date"]).rename(columns=str.lower)
K = []
for f in sorted(glob.glob("data/kaggle_patch/stocks/*.csv")):
    t = os.path.basename(f)[:-4]; k = pd.read_csv(f, parse_dates=["Date"]); a = k["Adj Close"] / k["Close"]
    K.append(pd.DataFrame({"ticker": t, "date": k.Date, "open": k.Open * a, "high": k.High * a, "low": k.Low * a, "close": k["Adj Close"], "volume": k.Volume, "src": "kaggle"}))
K = pd.concat(K); K = K[~K.ticker.isin(Y.ticker.unique())]; Y["src"] = "yahoo"
P = pd.concat([Y, K]).dropna(subset=["close"]); P = P[(P.close > 0) & (P.date >= "1999-01-01")].sort_values(["ticker", "date"]).reset_index(drop=True)
# ---- the shipped rules, tagged (mirrors adr_longshort/clean_prices.clean)
g = P.groupby("ticker").close; r1 = P.close / g.shift(1); r2 = g.shift(-1) / P.close
spike = (((r1 > 3) & (r2 < 1 / 3)) | ((r1 < 1 / 3) & (r2 > 3))).fillna(False)
Q = P[~spike].copy(); Q["ret"] = Q.groupby("ticker").close.pct_change()
fwd = pd.concat([Q.groupby("ticker")["ret"].shift(-k) for k in range(1, 6)], axis=1).max(axis=1); crash = ((Q.ret < -0.75) & (fwd > 3)).fillna(False)
reb_days = pd.Series(False, index=Q.index)
for k in range(1, 6):
    nxt = Q.groupby("ticker")["ret"].shift(-k); hit = crash & (nxt > 3); idx = Q.index[hit]; pos = [Q.index.get_loc(i) + k for i in idx]; reb_days.iloc[[p for p in pos if p < len(Q)]] = True
big = (Q.ret > 3).fillna(False) & ~reb_days & ~crash
ev = pd.concat([P.loc[spike, ["ticker", "date"]].assign(filter="R1 isolated print (row dropped)"),
                Q.loc[crash, ["ticker", "date"]].assign(filter="R2 crash day of a pair"), Q.loc[reb_days, ["ticker", "date"]].assign(filter="R2 rebound day of a pair"),
                Q.loc[big, ["ticker", "date"]].assign(filter="R3 single day > +300%")]).drop_duplicates(["ticker", "date"])
P["ret_raw"] = P.groupby("ticker").close.pct_change(); P["c_prev"] = P.groupby("ticker").close.shift(1); P["c_10"] = P.groupby("ticker").close.shift(-10)
P["vmed"] = P.groupby("ticker").volume.transform(lambda v: v.shift(1).rolling(20, min_periods=5).median()); P["vpost"] = P.groupby("ticker").volume.transform(lambda v: v.shift(-10).rolling(10, min_periods=5).median())
P["lr"] = np.log(P.close / P.c_prev)
flips = (P.lr.abs() > np.log(3)).astype(int); P["flips10"] = flips.groupby(P.ticker).transform(lambda s: s.rolling(21, center=True, min_periods=1).sum())
ev = ev.merge(P[["ticker", "date", "src", "open", "high", "low", "close", "volume", "ret_raw", "c_prev", "c_10", "vmed", "vpost", "lr", "flips10"]], on=["ticker", "date"], how="left")
S = pd.read_csv("research/data/splits_yf.csv", parse_dates=["date"])
ev["split_listed"] = [bool(((S.ticker == t) & ((S.date - d).abs() <= pd.Timedelta(days=7))).any()) for t, d in zip(ev.ticker, ev.date)]
# ---- holdings of each book: side and weight in the holding month
ev["hm"] = ev.date.dt.to_period("M")
for book, pred, mon_f in [("corrected", "research/out/p0_pred_corrected.csv", "research/out/p0_monthly_corrected.csv"), ("shipped", "results/predictions_rf.csv", "results/bt500k_rf_monthly.csv")]:
    F = load_predictions(pred); H = {}
    for ym, gg in F[F.liquid].groupby("ym"):
        h = holdings(gg)
        if h: H[ym + 1] = h
    mon = pd.read_csv(mon_f, parse_dates=["date"]); mon["hm"] = mon.date.dt.to_period("M") + 1; mm = mon.set_index("hm")
    ev[f"side_{book}"] = [("long" if t in H[m]["long"] else "short" if t in H[m]["short"] else "not held") if m in H else "not held" for t, m in zip(ev.ticker, ev.hm)]
    ev[f"pos_pct_{book}"] = [(mm.exposure.get(m, np.nan) / (len(H[m]["long"]) if sd == "long" else len(H[m]["short"])) * 100) if sd != "not held" else 0.0 for m, sd in zip(ev.hm, ev[f"side_{book}"])]
ev["side"] = np.where(ev.side_shipped != "not held", ev.side_shipped, ev.side_corrected)
# ---- second source: Nasdaq historical (listed names, ~10 years), fetched for held events and all events of listed names since 2016
def nasdaq(t):
    fp = f"research/data/nasdaq/{t}.json"
    if os.path.exists(fp): return json.load(open(fp))
    try:
        r = requests.get(f"https://api.nasdaq.com/api/quote/{t}/historical", params=dict(assetclass="stocks", fromdate="2014-01-01", todate="2026-10-10", limit=9999), headers={"User-Agent": "Mozilla/5.0"}, timeout=30); js = r.json()
    except Exception as e: js = {"error": str(e)}
    json.dump(js, open(fp, "w")); time.sleep(0.6); return js
need = ev[(ev.src == "yahoo") & ((ev.side != "not held") | (ev.date >= "2016-01-01"))].ticker.unique(); NQ = {}
for t in need:
    js = nasdaq(t); rows = (((js or {}).get("data") or {}).get("tradesTable") or {}).get("rows") or []
    if rows: NQ[t] = pd.Series({pd.Timestamp(x["date"]): float(x["close"].replace("$", "").replace(",", "")) for x in rows}).sort_index()
def second(t, d):
    s = NQ.get(t)
    if s is None or d not in s.index: return np.nan
    i = s.index.get_loc(d); return s.iloc[i] / s.iloc[i - 1] - 1 if i > 0 else np.nan
ev["ret_nasdaq"] = [second(t, d) for t, d in zip(ev.ticker, ev.date)]
# ---- classify
def lratio(a, b):
    return np.log(a / b) if pd.notna(a) and pd.notna(b) and a > 0 and b > 0 else np.nan
def cls(e):
    if e.split_listed: return "E1 split listed", "verified error"
    if pd.notna(e.ret_nasdaq) and abs(np.log1p(e.ret_nasdaq) - np.log1p(e.ret_raw)) > np.log(2): return "E4 second source disagrees", "verified error"
    if pd.notna(e.ret_nasdaq): return "R1 second source confirms", "real move"
    if abs(e.lr) > np.log(3) and np.sign(e.lr) * lratio(e.c_10, e.c_prev) > 0.6 * abs(e.lr) and np.sign(e.lr) * lratio(e.vpost, e.vmed) < -np.log(2): return "E2 split unlisted", "verified error"
    if e.flips10 >= 3: return "E3 mixed price scales", "verified error"
    if pd.notna(e.high) and pd.notna(e.low) and (e.close > e.high * 1.01 or e.close < e.low * 0.99): return "E5 close outside day's range", "verified error"
    if pd.notna(e.vmed) and e.vmed > 0 and e.volume / e.vmed >= 3 and pd.notna(e.high) and e.high >= e.close * 0.99: return "R1 volume confirms", "real move"
    return "U unknown", "unknown"
ev[["rule", "verdict"]] = ev.apply(lambda e: pd.Series(cls(e)), axis=1)
ev["direction"] = np.where(ev.ret_raw > 0, "up", "down")
ev.to_csv("research/out/p0_audit_events.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(f"events {len(ev)}; by filter:\n{ev['filter'].value_counts().to_string()}\nverdicts:\n{ev.groupby(['verdict','rule']).size().to_string()}")
A = ev.pivot_table(index=["side_shipped", "direction"], columns="verdict", values="ticker", aggfunc="count", fill_value=0); A["total"] = A.sum(axis=1); A.to_csv("research/out/p0_audit_asymmetry.csv"); print("\nasymmetry (all events):\n", A.to_string())
held = ev[ev.side != "not held"]; print(f"\nheld events ({len(held)}):\n", held[["ticker", "date", "side_shipped", "side_corrected", "filter", "rule", "verdict", "ret_raw", "ret_nasdaq", "pos_pct_shipped"]].round(3).to_string(index=False))
# ---- restored P&L: only verified errors are removed
err = ev[ev.verdict == "verified error"]; drop = set(zip(err[err["filter"].str.startswith("R1")].ticker, err[err["filter"].str.startswith("R1")].date))
nullk = set(zip(err[~err["filter"].str.startswith("R1")].ticker, err[~err["filter"].str.startswith("R1")].date))
base = P[["ticker", "date", "close"]].copy(); keep = [(t, d) not in drop for t, d in zip(base.ticker, base.date)]; base = base[keep].copy(); base["ret"] = base.groupby("ticker").close.pct_change()
base.loc[[(t, d) in nullk for t, d in zip(base.ticker, base.date)], "ret"] = np.nan
RET_rest = base[base.date >= "2009-12-01"].pivot(index="date", columns="ticker", values="ret")
rows = []
RET_raw = P[P.date >= "2009-12-01"].pivot(index="date", columns="ticker", values="ret_raw")
for tag, pred, ro in [("corrected, shipped cleaning", "research/out/p0_pred_corrected.csv", None), ("corrected, only verified errors removed", "research/out/p0_pred_corrected.csv", RET_rest), ("corrected, no cleaning", "research/out/p0_pred_corrected.csv", RET_raw), ("shipped book, shipped cleaning", "results/predictions_rf.csv", None), ("shipped book, only verified errors removed", "results/predictions_rf.csv", RET_rest), ("shipped book, no cleaning", "results/predictions_rf.csv", RET_raw)]:
    ns = load_ns(pred, ret_override=ro); d, m = ns["simulate"](); s = ns["stats"](d, m)
    rows.append(dict(variant=tag, final=s["final_equity"], sharpe=s["sharpe"], sharpe_m=s["sharpe_monthly"], maxdd=s["maxdd_pct"], worst_month=s["worst_month_pct"], worst_day=d.ret.min() * 100, worst_day_date=str(d.ret.idxmin().date()), peak_maint=s["peak_maint_pct"], call_days=s["call_days"]))
    log("P0", f"restored-P&L test: {tag}", f"sharpe {s['sharpe']:.2f} worst month {s['worst_month_pct']:.1f}% call days {s['call_days']}", "research/out/p0_restored_accounts.csv")
T = pd.DataFrame(rows); T.to_csv("research/out/p0_restored_accounts.csv", index=False); print("\n", T.round(2).to_string(index=False))
