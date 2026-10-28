"""Fetch SEC fails-to-deliver (all semi-monthly zips) and Nasdaq Reg SHO threshold lists (every business day 2010-2026).
Keeps only rows for universe/held tickers. Raw files are not kept (filtered CSVs are). Usage: ... ftd|regsho"""
import io, os, re, sys, time, zipfile, urllib.request, pandas as pd
from concurrent.futures import ThreadPoolExecutor
U = set(pd.read_csv("data/universe.csv").ticker) | set(pd.read_csv("research/out/p4_held.csv").ticker)
D = "research/data/borrow"; what = sys.argv[1]
def get(url, ua):
    for k in range(3):
        try: return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": ua}), timeout=60).read()
        except Exception as e:
            if "404" in str(e): return None
            time.sleep(2 + 3 * k)
    return None
if what == "ftd":
    UA = "adr_longshort research bot (academic)"
    page = get("https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data", UA).decode("utf8", "ignore")
    links = sorted(set(re.findall(r'href="(/files/(?:data|node/add/data_distribution)/[^"]*cnsfails\d{6}[ab]\.zip)"', page))); out = []; log = []   # Feb-Apr 2020 sit under /files/node/add/
    for l in links:
        b = get("https://www.sec.gov" + l, UA); time.sleep(0.25)
        if b is None: log.append((l, "fail")); continue
        try:
            z = zipfile.ZipFile(io.BytesIO(b)); txt = z.read(z.namelist()[0]).decode("latin1")
            df = pd.read_csv(io.StringIO(txt), sep="|", dtype=str, on_bad_lines="skip"); df.columns = [c.strip().upper() for c in df.columns]
            df = df[df["SYMBOL"].isin(U)]; out.append(df); log.append((l, len(df)))
        except Exception as e: log.append((l, f"parse {e}"[:80]))
    pd.concat(out).to_csv(f"{D}/ftd_universe.csv", index=False); pd.DataFrame(log, columns=["file", "rows"]).to_csv(f"{D}/ftd_log.csv", index=False); print("ftd files", len(links))
if what == "regsho":
    UA = "Mozilla/5.0 (adr_longshort research)"; days = pd.bdate_range("2010-01-01", "2026-10-16")
    def one(d):
        b = get(f"https://www.nasdaqtrader.com/dynamic/symdir/regsho/nasdaqth{d:%Y%m%d}.txt", UA)
        if b is None: return (d, None)
        rows = [ln.split("|") for ln in b.decode("latin1").splitlines()[1:] if "|" in ln]
        return (d, [r[0] for r in rows if len(r) > 3 and r[3].strip() == "Y"] if rows else [])
    with ThreadPoolExecutor(4) as ex: res = list(ex.map(one, days))
    got = [(d, s) for d, s in res if s is not None]; rec = [(d.date(), s) for d, ss in got for s in ss if s in U]
    pd.DataFrame(rec, columns=["date", "ticker"]).to_csv(f"{D}/regsho_nasdaq_universe.csv", index=False)
    pd.DataFrame([(d.date(), s is not None, len(s or [])) for d, s in res], columns=["date", "file_ok", "n_all"]).to_csv(f"{D}/regsho_log.csv", index=False)
    print("regsho days ok", len(got), "of", len(days))
