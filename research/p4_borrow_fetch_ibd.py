"""Fetch iBorrowDesk JSON (IBKR fee/availability, last ~12 months daily) for every held ticker; cache raw JSON. ~1 req/sec."""
import os, time, json, urllib.request, pandas as pd
H = pd.read_csv("research/out/p4_held.csv"); tick = list(dict.fromkeys(H.sort_values("side", ascending=False).ticker))   # shorts first
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"; log = []
for t in tick:
    f = f"research/data/borrow/ibd/{t}.json"
    if os.path.exists(f): continue
    try:
        r = urllib.request.urlopen(urllib.request.Request(f"https://www.iborrowdesk.com/api/ticker/{t}", headers={"User-Agent": UA, "Accept": "application/json"}), timeout=30)
        b = r.read(); open(f, "wb").write(b); log.append((t, r.status, len(b)))
    except Exception as e: log.append((t, str(e)[:60], 0)); open(f, "w").write(json.dumps({"error": str(e)[:200]}))
    time.sleep(1.0)
pd.DataFrame(log, columns=["ticker", "status", "bytes"]).to_csv("research/data/borrow/ibd_fetch_log.csv", index=False); print("done", len(log))
