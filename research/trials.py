"""Trial log: one row per variant run. research/out/trial_log_<phase>.csv (phases log to separate files; merged at report time)."""
import os, csv, datetime
def log(phase, description, metric_summary, file=""):
    p = f"research/out/trial_log_{phase.lower()}.csv"; new = not os.path.exists(p)
    with open(p, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["trial_id", "phase", "timestamp", "description", "metric_summary", "file"])
        n = sum(1 for _ in open(p)) if not new else 1
        w.writerow([f"{phase}-{n:03d}", phase, datetime.datetime.now().isoformat(timespec="seconds"), description, metric_summary, file])
