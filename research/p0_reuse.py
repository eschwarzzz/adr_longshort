"""Phase 0.6 add-on: drop each name's prices before Jan 1 of (IPO year - 1) (ticker reuse / pre-merger shells), then corrected baseline."""
import sys, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "research")
from adr_longshort import load_prices
from corrected import build, refit_predict
from metrics import full_report
from trials import log
P = load_prices(); U = pd.read_csv("data/universe.csv"); ipo = pd.to_numeric(U.set_index("ticker").ipo_year, errors="coerce")
cut = P.ticker.map(ipo).sub(1).fillna(1900).astype(int); keep = P.date.dt.year >= cut; print(f"rows dropped: {(~keep).sum():,} from {P[~keep].ticker.nunique()} names")
M, _ = build(P=P[keep].reset_index(drop=True)); Pr, hp = refit_predict(M); Pr.to_csv("research/out/p0_pred_corrected_reuse.csv", index=False); M.to_csv("research/out/p0_panel_corrected_reuse.csv", index=False)
o, _ = full_report("research/out/p0_pred_corrected_reuse.csv"); b, _ = full_report("research/out/p0_pred_corrected.csv")
T = pd.DataFrame({"corrected": b, "corrected + ticker-reuse fix": o}); T.to_csv("research/out/p0_reuse_compare.csv"); print(T.round(3).to_string())
log("P0", "corrected + ticker-reuse truncation", f"acct sharpe {o['acct_sharpe']:.2f} unit {o['unit_sharpe']:.2f}", "research/out/p0_reuse_compare.csv")
