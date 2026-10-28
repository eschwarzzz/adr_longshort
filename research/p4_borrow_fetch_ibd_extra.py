"""iBorrowDesk for tickers in the three forecast files not yet cached (curl-free urllib, 3 s apart, stop on a 444 block)."""
import os, glob, time, json, urllib.request, pandas as pd
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
have = {os.path.basename(f)[:-5] for f in glob.glob("research/data/borrow/ibd/*.json")}
tick = set()
for f in ["results/predictions_rf.csv", "research/out/p0_pred_corrected.csv", "research/out/p0_pred_corrected_reuse.csv"]: tick |= set(pd.read_csv(f, usecols=["ticker"]).ticker)
todo = sorted(tick - have); print("to fetch", len(todo), todo, flush=True); n = 0
for t in todo:
    try: b = urllib.request.urlopen(urllib.request.Request(f"https://www.iborrowdesk.com/api/ticker/{t}", headers={"User-Agent": UA, "Accept": "application/json"}), timeout=30).read(); open(f"research/data/borrow/ibd/{t}.json", "wb").write(b); n += 1
    except Exception as e:
        if "444" in str(e): print("blocked at", t, flush=True); break
        open(f"research/data/borrow/ibd/{t}.json", "w").write(json.dumps({"error": str(e)[:200]}))
    time.sleep(3)
print("fetched", n, flush=True)
