"""Slow retry of iBorrowDesk tickers that failed with HTTP 444 (rate limit). Waits until the API answers again (probe every 2 min, up to 90 min),
then 8 s between requests; on a new 444 it waits for the block to lift again. Order: names held in holding months >= 2025-10 first."""
import os, glob, time, json, urllib.request, pandas as pd
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
def fetch(t):
    return urllib.request.urlopen(urllib.request.Request(f"https://www.iborrowdesk.com/api/ticker/{t}", headers={"User-Agent": UA, "Accept": "application/json"}), timeout=30).read()
def wait_open(limit=5400):
    t0 = time.time()
    while time.time() - t0 < limit:
        try: fetch("JD"); return True
        except Exception: time.sleep(120)
    return False
H = pd.read_csv("research/out/p4_held.csv"); recent = set(H[H.hold_month >= "2025-10"].ticker); shorts = set(H[H.side == "short"].ticker)
bad = [os.path.basename(f)[:-5] for f in glob.glob("research/data/borrow/ibd/*.json") if '"error": "HTTP Error 444' in open(f).read()[:60]]
bad.sort(key=lambda t: (t not in recent, t not in shorts, t)); print("to retry", len(bad), "recent-held", sum(t in recent for t in bad), flush=True)
if not wait_open(): print("API still blocked after 90 min; stopping", flush=True); raise SystemExit
done = 0
for t in bad:
    for k in range(3):
        try: open(f"research/data/borrow/ibd/{t}.json", "wb").write(fetch(t)); done += 1; break
        except Exception as e:
            if "444" in str(e) or "429" in str(e):
                print("blocked again at", t, done, flush=True)
                if not wait_open(): print("stopping; fetched", done, flush=True); raise SystemExit
            elif "404" in str(e): open(f"research/data/borrow/ibd/{t}.json", "w").write(json.dumps({"error": "404 not found"})); break
            else: time.sleep(10)
    time.sleep(8)
print("retry finished; fetched", done, flush=True)
