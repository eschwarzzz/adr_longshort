"""Phase 0.7b: data.sec.gov submissions for every Form 25 filer (cached compactly) -> research/data/p07/subs/<cik>.json"""
import os, json, time, requests, pandas as pd
UA = {"User-Agent": "adr_longshort academic research admin@example.org"}
os.makedirs("research/data/p07/subs", exist_ok=True)
D = pd.read_csv("research/data/p07/form25.csv"); ciks = sorted(D.cik.unique()); KEEP = {"SC 13E3", "SC 13E3/A", "DEFM14A", "SC TO-T", "SC 14D9", "15-12B", "15-12G", "15F-12B", "15F-12G", "F-1", "S-1", "424B4", "424B1", "20-F", "10-K", "NT 20-F", "NT 10-K", "8-K", "6-K", "F-4", "S-4", "25", "25-NSE"}
s = requests.Session(); s.headers.update(UA); t0 = time.time(); n = 0; fails = 0
for i, c in enumerate(ciks):
    fp = f"research/data/p07/subs/{c}.json"
    if os.path.exists(fp): continue
    for attempt in range(4):
        try:
            r = s.get(f"https://data.sec.gov/submissions/CIK{c:010d}.json", timeout=30)
            if r.status_code == 429 or r.status_code >= 500: time.sleep(5 * (attempt + 1)); continue
            break
        except Exception: time.sleep(3)
    else: fails += 1; continue
    if r.status_code != 200: json.dump({"status": r.status_code}, open(fp, "w")); continue
    j = r.json(); f = j.get("filings", {}).get("recent", {})
    forms = [(fm, d) for fm, d in zip(f.get("form", []), f.get("filingDate", [])) if fm in KEEP]
    out = {k: j.get(k) for k in ("name", "tickers", "exchanges", "sic", "sicDescription", "stateOfIncorporation", "formerNames", "entityType")}
    out["biz"] = (j.get("addresses") or {}).get("business", {}).get("stateOrCountry"); out["mail"] = (j.get("addresses") or {}).get("mailing", {}).get("stateOrCountry")
    out["forms"] = forms; out["n_recent"] = len(f.get("form", [])); out["older_files"] = len(j.get("filings", {}).get("files", []))
    json.dump(out, open(fp, "w")); n += 1
    time.sleep(0.13)
    if n % 500 == 0: print(f"{i+1}/{len(ciks)} [{time.time()-t0:.0f}s]", flush=True)
print("done", n, "fetched, fails", fails)
