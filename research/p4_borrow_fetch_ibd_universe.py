"""Second pass: iBorrowDesk for every universe ticker with Yahoo prices (to widen the fee-model training cross-section). Skips cached."""
import os, time, json, gzip, urllib.request, pandas as pd
tick = sorted(pd.read_csv("data/yf_prices.csv.gz", usecols=["ticker"]).ticker.unique())
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"; n = 0
for t in tick:
    f = f"research/data/borrow/ibd/{t}.json"
    if os.path.exists(f): continue
    try: b = urllib.request.urlopen(urllib.request.Request(f"https://www.iborrowdesk.com/api/ticker/{t}", headers={"User-Agent": UA, "Accept": "application/json"}), timeout=30).read(); open(f, "wb").write(b)
    except Exception as e: open(f, "w").write(json.dumps({"error": str(e)[:200]}))
    n += 1; time.sleep(1.0)
print("universe pass fetched", n)
