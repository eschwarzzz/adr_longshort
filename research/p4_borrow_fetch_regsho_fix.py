"""Refetch Nasdaq threshold-list days that came back empty (bot-protection pages served as 200), sequentially, accepting only files whose first line
is the real header. Patches regsho_nasdaq_universe.csv and regsho_log.csv (adds column verified)."""
import time, urllib.request, pandas as pd
D = "research/data/borrow"; U = set(pd.read_csv("data/universe.csv").ticker) | set(pd.read_csv("research/out/p4_held.csv").ticker)
L = pd.read_csv(f"{D}/regsho_log.csv", parse_dates=["date"]); R = pd.read_csv(f"{D}/regsho_nasdaq_universe.csv", parse_dates=["date"])
L["verified"] = L.n_all > 0; todo = L[L.file_ok & (L.n_all == 0)].date; print("refetch", len(todo), flush=True); new = []
for d in todo:
    ok = False
    for k in range(4):
        try:
            b = urllib.request.urlopen(urllib.request.Request(f"https://www.nasdaqtrader.com/dynamic/symdir/regsho/nasdaqth{d:%Y%m%d}.txt", headers={"User-Agent": "Mozilla/5.0 (adr_longshort research)"}), timeout=60).read().decode("latin1")
            if b.startswith("Symbol|"):
                rows = [ln.split("|") for ln in b.splitlines()[1:] if "|" in ln]; ys = [r[0] for r in rows if len(r) > 3 and r[3].strip() == "Y"]
                L.loc[L.date == d, ["n_all", "verified"]] = [len(ys), True]; new += [(d, s) for s in ys if s in U]; ok = True; break
        except Exception: pass
        time.sleep(5 * (k + 1))
    time.sleep(0.7)
R = pd.concat([R, pd.DataFrame(new, columns=["date", "ticker"])]).drop_duplicates(); R.to_csv(f"{D}/regsho_nasdaq_universe.csv", index=False); L.to_csv(f"{D}/regsho_log.csv", index=False)
print("verified days", int(L.verified.sum()), "of", len(L), "| new universe rows", len(new), flush=True)
