"""Wayback Machine snapshots of iBorrowDesk's API JSON for held short tickers (each snapshot carries ~12 months of daily IBKR fee history).
Keeps at most one snapshot per ticker per half-year. Raw JSON cached under research/data/borrow/ibd_wayback/<T>_<timestamp>.json."""
import os, time, json, urllib.request, pandas as pd
H = pd.read_csv("research/out/p4_held.csv"); tick = sorted(set(H[H.side == "short"].ticker)); log = []
def get(u, t=60):
    for k in range(3):
        try: return urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "adr_longshort research (academic)"}), timeout=t).read()
        except Exception: time.sleep(3 * (k + 1))
    return None
for t in tick:
    snaps = set()
    for host in ("iborrowdesk.com", "www.iborrowdesk.com"):
        b = get(f"http://web.archive.org/cdx/search/cdx?url={host}/api/ticker/{t}&output=txt&fl=timestamp,statuscode&filter=statuscode:200")
        if b: snaps |= {ln.split()[0] for ln in b.decode().splitlines() if ln.strip()}
        time.sleep(1)
    keep = {}
    for s in sorted(snaps): keep.setdefault(s[:4] + ("a" if s[4:6] <= "06" else "b"), s)
    got = 0
    for s in keep.values():
        f = f"research/data/borrow/ibd_wayback/{t}_{s}.json"
        if os.path.exists(f): got += 1; continue
        b = get(f"http://web.archive.org/web/{s}id_/https://iborrowdesk.com/api/ticker/{t}", 90)
        if b and b.strip().startswith(b"{"): open(f, "wb").write(b); got += 1
        time.sleep(1.5)
    log.append((t, len(snaps), len(keep), got))
pd.DataFrame(log, columns=["ticker", "snapshots", "kept", "downloaded"]).to_csv("research/data/borrow/wayback_log.csv", index=False)
print("wayback done", sum(r[3] for r in log), "files for", sum(r[3] > 0 for r in log), "tickers", flush=True)
