"""Phase 0.7a: Form 25 / 25-NSE filings 2009-2026 from SEC EDGAR full-index form.idx -> research/data/p07/form25.csv"""
import os, time, re, requests, pandas as pd
UA = {"User-Agent": "adr_longshort academic research admin@example.org"}
rows = []
for y in range(2009, 2027):
    for q in range(1, 5):
        fp = f"research/data/p07/form_{y}Q{q}.idx"
        if not os.path.exists(fp):
            r = requests.get(f"https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/form.idx", headers=UA, timeout=60)
            if r.status_code != 200: print(y, q, r.status_code); continue
            open(fp, "wb").write(r.content); time.sleep(0.2)
        for line in open(fp, encoding="latin-1"):
            if line.startswith("25 ") or line.startswith("25-NSE ") or line.startswith("25/A ") or line.startswith("25-NSE/A "):
                m = re.match(r"^(\S+)\s+(.*?)\s+(\d+)\s+(\d{4}-\d{2}-\d{2})\s+(\S+)\s*$", line)
                if m: rows.append(dict(form=m.group(1), company=m.group(2), cik=int(m.group(3)), date=m.group(4), file=m.group(5)))
D = pd.DataFrame(rows); D.to_csv("research/data/p07/form25.csv", index=False)
print(len(D), "filings,", D.cik.nunique(), "CIKs"); print(D.form.value_counts().to_string())
