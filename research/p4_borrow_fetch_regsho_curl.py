"""Reg SHO threshold lists via curl (the sites serve bot-challenge pages to urllib). Accepts only files whose first line is the real header.
  nasdaq: refetch every day not yet verified (2010-2026, all business days)
  nyse:   NYSE and NYSE American, every 10th business day 2010-2026 (Cloudflare rate limit; 2.5 s per request)
Writes regsho_nasdaq_universe.csv / regsho_log.csv (verified column) and regsho_nyse_universe.csv / regsho_nyse_log.csv."""
import sys, time, subprocess, urllib.parse, pandas as pd
D = "research/data/borrow"; U = set(pd.read_csv("data/universe.csv").ticker) | set(pd.read_csv("research/out/p4_held.csv").ticker); which = sys.argv[1]
def blocked(t): return "1015" in t[:100] or "_Incapsula_" in t[:400]
def wait_open(url, limit=5400):
    t0 = time.time()
    while time.time() - t0 < limit:
        r = subprocess.run(["curl", "-s", "-m", "40", "-A", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", url], capture_output=True).stdout.decode("latin1")
        if not blocked(r): return True
        time.sleep(120)
    return False
PAUSE = 0.4
def curl(url):
    for k in range(3):
        r = subprocess.run(["curl", "-s", "-m", "40", "-A", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", url], capture_output=True)
        t = r.stdout.decode("latin1")
        if t.startswith("Symbol|"): return t
        if blocked(t):
            print("blocked; waiting", flush=True)
            if not wait_open(url): print("still blocked, giving up", flush=True); raise SystemExit
            continue
        if "Error" in t[:200] or "404" in t[:200]: return None
        time.sleep(3 * (k + 1))
    return None
def parse(t): return [ln.split("|")[0] for ln in t.splitlines()[1:] if "|" in ln and len(ln.split("|")) > 3 and ln.split("|")[3].strip() == "Y"]
if which == "nasdaq":
    L = pd.read_csv(f"{D}/regsho_log.csv", parse_dates=["date"]); R = pd.read_csv(f"{D}/regsho_nasdaq_universe.csv", parse_dates=["date"])
    if "verified" not in L: L["verified"] = L.n_all > 0
    todo = list(L[~L.verified.astype(bool)].date); print("nasdaq refetch", len(todo), flush=True); new = []
    for i, d in enumerate(todo):
        t = curl(f"https://www.nasdaqtrader.com/dynamic/symdir/regsho/nasdaqth{d:%Y%m%d}.txt")
        if t is not None:
            ys = parse(t); L.loc[L.date == d, ["n_all", "verified", "file_ok"]] = [len(ys), True, True]; new += [(d, s) for s in ys if s in U]
        time.sleep(0.4)
        if i % 100 == 0: print(i, flush=True)
    R = pd.concat([R, pd.DataFrame(new, columns=["date", "ticker"])]).drop_duplicates(); R.to_csv(f"{D}/regsho_nasdaq_universe.csv", index=False); L.to_csv(f"{D}/regsho_log.csv", index=False)
    print("nasdaq verified days", int(L.verified.sum()), "of", len(L), flush=True)
if which == "nyse":
    days = pd.bdate_range("2010-01-04", "2026-10-16")[::10]; rec = []; lg = []
    for i, d in enumerate(days):
        for mk in ("NYSE", "NYSE American"):
            t = curl(f"https://www.nyse.com/api/regulatory/threshold-securities/download?selectedDate={d:%Y-%m-%d}&market={urllib.parse.quote(mk)}")
            ys = parse(t) if t else []; lg.append((d.date(), mk, t is not None, len(ys))); rec += [(d.date(), mk, s) for s in ys if s in U]; time.sleep(2.5)
        if i % 100 == 0: print(i, flush=True)
    pd.DataFrame(rec, columns=["date", "market", "ticker"]).to_csv(f"{D}/regsho_nyse_universe.csv", index=False); pd.DataFrame(lg, columns=["date", "market", "ok", "n_all"]).to_csv(f"{D}/regsho_nyse_log.csv", index=False)
    print("nyse ok", sum(x[2] for x in lg), "of", len(lg), flush=True)
