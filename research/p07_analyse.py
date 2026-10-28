"""Phase 0.7c: China/HK issuers that delisted from US exchanges 2009-2026, by year and type; coverage in our universe and price data;
the post-April-2020 micro-IPO graveyard; plausible pool membership by analogy with surviving names.
Classification (fixed in advance):
  final delisting  = a Form 25/25-NSE after which the issuer files no 10-K/20-F more than 400 days later (else exchange transfer / other security)
  take-private/M&A = SC 13E3, DEFM14A, SC TO-T, SC 14D9, F-4 or S-4 within 540 days before the delisting
  voluntary        = issuer-filed Form 25, no M&A filing
  exchange-initiated = 25-NSE only, no M&A filing (non-compliance, late filings, fraud); late_filer flag = NT 10-K/NT 20-F within 2 years before"""
import os, json, re, glob, numpy as np, pandas as pd
D = pd.read_csv("research/data/p07/form25.csv", parse_dates=["date"]); rows = []
CN = {"F4", "K3"}; MA = {"SC 13E3", "SC 13E3/A", "DEFM14A", "SC TO-T", "SC 14D9", "F-4", "S-4"}
for c, g in D.groupby("cik"):
    fp = f"research/data/p07/subs/{c}.json"
    if not os.path.exists(fp): continue
    j = json.load(open(fp))
    if "status" in j: continue
    if not ({j.get("biz"), j.get("mail"), j.get("stateOfIncorporation")} & CN): continue
    forms = [(f, pd.Timestamp(d)) for f, d in j["forms"]]; ann = [d for f, d in forms if f in ("10-K", "20-F")]
    ipo = min([d for f, d in forms if f in ("424B4", "424B1", "F-1", "S-1")], default=pd.NaT)
    last_any = max([d for f, d in forms], default=pd.NaT)
    for d in sorted(g.date.unique()):
        d = pd.Timestamp(d)
        if any(a > d + pd.Timedelta(days=400) for a in ann): continue          # still reporting long after: transfer or another security
        fm = set(g[g.date == d].form); win = [f for f, x in forms if d - pd.Timedelta(days=540) <= x <= d]
        typ = "take-private/M&A" if set(win) & MA else ("voluntary" if any(f.startswith("25") and not f.startswith("25-NSE") for f in fm) or any(f in ("25", "25/A") for f, x in forms if abs((x - d).days) < 30) else "exchange-initiated")
        late = any(f in ("NT 10-K", "NT 20-F") for f, x in forms if d - pd.Timedelta(days=730) <= x <= d)
        rows.append(dict(cik=c, name=j["name"], former=";".join(x.get("name", "") for x in (j.get("formerNames") or [])), tickers=";".join(j.get("tickers") or []), biz=j.get("biz"), inc=j.get("stateOfIncorporation"),
                         sic=j.get("sic"), delist_date=d, type=typ, late_filer=late, ipo_filing=ipo, older_files=j.get("older_files", 0)))
        break                                                                 # first final delisting per issuer
X = pd.DataFrame(rows)
# exclude funds/trusts/SPAC shells by SIC 6770 (blank checks) reported separately
X["spac"] = X.sic.astype(str).eq("6770"); X["year"] = X.delist_date.dt.year
# coverage: universe.csv and price data (by name, then ticker)
U = pd.read_csv("data/universe.csv"); norm = lambda s: re.sub(r"[^a-z0-9 ]", "", str(s).lower()).replace("  ", " ").split()
def key(s): w = [x for x in norm(s) if x not in ("the", "inc", "ltd", "limited", "co", "corp", "corporation", "holdings", "holding", "group", "plc", "company", "international")]; return " ".join(w[:2])
U["k"] = U.name.map(key); uk = dict(zip(U.k, U.ticker))
import sys; sys.path.insert(0, "."); from adr_longshort import load_prices
P = load_prices(); last = P.groupby("ticker").date.max(); first = P.groupby("ticker").date.min()
def uni(r):
    for nm in [r["name"]] + [x for x in r["former"].split(";") if x]:
        if key(nm) in uk: return uk[key(nm)]
    for t in r["tickers"].split(";"):
        if t and t in set(U.ticker): return t
    return ""
X["universe_ticker"] = X.apply(uni, axis=1); X["in_universe"] = X.universe_ticker.ne("")
X["prices_cover_delisting"] = [bool(t) and t in last.index and last[t] >= d - pd.Timedelta(days=45) and first[t] <= d for t, d in zip(X.universe_ticker, X.delist_date)]
X["has_any_prices"] = [bool(t) and t in last.index for t in X.universe_ticker]
X["graveyard"] = (X.ipo_filing >= "2020-04-01") & (X.delist_date < "2026-10-01")
X.to_csv("research/out/p07_delisted_china_hk.csv", index=False)
E = X[~X.spac]
print(f"China/HK-based issuers with a final US delisting 2009-2026: {len(X)} (SPAC shells {X.spac.sum()}); operating: {len(E)}")
T = E.pivot_table(index="year", columns="type", values="cik", aggfunc="count", fill_value=0); T["total"] = T.sum(axis=1)
T["late_filer"] = E.groupby("year").late_filer.sum(); T["in_universe"] = E.groupby("year").in_universe.sum(); T["prices_cover"] = E.groupby("year").prices_cover_delisting.sum()
print(T.to_string()); T.to_csv("research/out/p07_by_year.csv")
G = E[E.graveyard]; print(f"\ngraveyard (IPO filing after Apr 2020, delisted before Oct 2026): {len(G)}; by type {G.type.value_counts().to_dict()}; in universe {G.in_universe.sum()}; with prices {G.has_any_prices.sum()}")
pre = E[E.delist_date < "2020-04-01"]; print(f"delisted before Apr 2020: {len(pre)}; in universe {pre.in_universe.sum()}; prices covering delisting {pre.prices_cover_delisting.sum()}; by type {pre.type.value_counts().to_dict()}")
mid = E[(E.delist_date >= "2020-04-01")]; print(f"delisted Apr 2020 or later: {len(mid)}; in universe {mid.in_universe.sum()}; prices covering delisting {mid.prices_cover_delisting.sum()}; by type {mid.type.value_counts().to_dict()}")
