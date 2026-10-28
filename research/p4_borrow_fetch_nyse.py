"""NYSE and NYSE American Reg SHO threshold lists, every 5th business day 2010-2026 (threshold status needs 5 consecutive settlement days of fails,
so a 5-day sample catches every spell); universe tickers only."""
import time, urllib.request, urllib.parse, pandas as pd
from concurrent.futures import ThreadPoolExecutor
U = set(pd.read_csv("data/universe.csv").ticker) | set(pd.read_csv("research/out/p4_held.csv").ticker)
days = pd.bdate_range("2010-01-04", "2026-10-16")[::5]
def one(a):
    d, mk = a
    for k in range(3):
        try:
            b = urllib.request.urlopen(urllib.request.Request(f"https://www.nyse.com/api/regulatory/threshold-securities/download?selectedDate={d:%Y-%m-%d}&market={urllib.parse.quote(mk)}", headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read().decode("latin1")
            return (d.date(), mk, True, [ln.split("|")[0] for ln in b.splitlines()[1:] if "|" in ln])
        except Exception: time.sleep(2)
    return (d.date(), mk, False, [])
with ThreadPoolExecutor(4) as ex: res = list(ex.map(one, [(d, m) for d in days for m in ("NYSE", "NYSE American")]))
pd.DataFrame([(d, mk, s) for d, mk, ok, ss in res for s in ss if s in U], columns=["date", "market", "ticker"]).to_csv("research/data/borrow/regsho_nyse_universe.csv", index=False)
pd.DataFrame([(d, mk, ok, len(ss)) for d, mk, ok, ss in res], columns=["date", "market", "ok", "n_all"]).to_csv("research/data/borrow/regsho_nyse_log.csv", index=False); print("nyse done", sum(r[2] for r in res), len(res))
