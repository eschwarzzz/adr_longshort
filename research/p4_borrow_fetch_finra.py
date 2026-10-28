"""FINRA consolidated short interest bi-monthly files (cdn.finra.org, exchange-listed included, ~2017+). Finds each half-month's
settlement-date file by trying business days, keeps universe/held tickers."""
import io, time, urllib.request, pandas as pd
U = set(pd.read_csv("data/universe.csv").ticker) | set(pd.read_csv("research/out/p4_held.csv").ticker)
def get(d):
    try: return urllib.request.urlopen(urllib.request.Request(f"https://cdn.finra.org/equity/otcmarket/biweekly/shrt{d:%Y%m%d}.csv", headers={"User-Agent": "Mozilla/5.0 (adr_longshort research)"}), timeout=60).read()
    except Exception: return None
out = []; log = []
for m in pd.period_range("2017-12", "2026-10", freq="M"):
    s = m.start_time
    for lo, hi in ((8, 17), (23, 31)):
        cands = [d for d in pd.bdate_range(s + pd.Timedelta(days=lo - 1), min(s + pd.Timedelta(days=hi - 1), m.end_time))][::-1]
        hit = None
        for d in cands:
            b = get(d); time.sleep(0.1)
            if b and b[:200].decode("latin1", "ignore").startswith("accountingYearMonthNumber"): hit = d; break
        if hit is None: log.append((str(m), lo, None, 0)); continue
        df = pd.read_csv(io.BytesIO(b), sep="|", dtype=str, quoting=3, on_bad_lines="skip"); df = df[df.symbolCode.isin(U)]; out.append(df); log.append((str(m), lo, hit.date(), len(df)))
pd.concat(out).to_csv("research/data/borrow/finra_si_universe.csv", index=False); pd.DataFrame(log, columns=["month", "half", "file_date", "rows"]).to_csv("research/data/borrow/finra_log.csv", index=False)
print("finra files", sum(1 for r in log if r[2]))
