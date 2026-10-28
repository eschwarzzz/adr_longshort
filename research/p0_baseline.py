"""Phase 0.6 (part 1): build the corrected panels and forecasts, one fix at a time and all together.
Writes research/out/p0_panel_<tag>.csv (all) and research/out/p0_pred_<tag>.csv per variant, and logs trials."""
import sys, time, pandas as pd
sys.path.insert(0, "research"); from corrected import build, refit_predict
from trials import log
V = {"shipped": dict(real_price=False, no_cap=False, median_adv=False, next_filter="shipped"),
     "fix_price": dict(real_price=True, no_cap=True, median_adv=False, next_filter="shipped"),
     "fix_next": dict(real_price=False, no_cap=False, median_adv=False, next_filter="truncation"),
     "fix_adv": dict(real_price=False, no_cap=False, median_adv=True, next_filter="shipped"),
     "corrected": dict(real_price=True, no_cap=True, median_adv=True, next_filter="truncation")}
for tag, kw in V.items():
    t0 = time.time(); M, st = build(**kw); P, hp = refit_predict(M)
    if tag == "corrected": M.to_csv("research/out/p0_panel_corrected.csv", index=False)
    P.to_csv(f"research/out/p0_pred_{tag}.csv", index=False); hp.to_csv(f"research/out/p0_hyper_{tag}.csv", index=False)
    msg = f"rows scored {len(P):,}, liquid {int(P.liquid.sum()):,}, names {P.ticker.nunique()}"
    log("P0", f"baseline variant {tag}: {kw}", msg, f"research/out/p0_pred_{tag}.csv"); print(tag, msg, f"[{time.time()-t0:.0f}s]", flush=True)
