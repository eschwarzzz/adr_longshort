"""Charts for the $500,000 account backtest (random forest). Full window: fig2 equity and drawdown, fig3 signal / volatility / exposure / margin, fig4 Monte Carlo.
Last five years (fresh account from Sep 2021): fig5 account page, fig6 Monte Carlo, both also in results/bt500k_rf_5y_pages.pdf.

    python3 make_charts.py
"""
import pandas as pd, numpy as np, matplotlib, json, os
matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib.ticker import FuncFormatter; from matplotlib.backends.backend_pdf import PdfPages
from adr_longshort import ROOT
R = f"{ROOT}/results"; MODEL = "RF"; ALT = "Ensemble"; TAG = "bt500k_rf"; SUF = "_rf"
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"; S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e6bc7"]; BAND = ["#cde2fb", "#9ec5f4"]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"], "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
def style(ax):
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=1); ax.set_axisbelow(True); ax.tick_params(length=0)
def title(ax, t, size=11): ax.set_title(t, loc="left", fontsize=size, fontweight="semibold", pad=8)
kusd = FuncFormatter(lambda v, _: f"${v/1e6:,.1f}m" if v >= 1e6 else f"${v/1000:,.0f}k"); pct = FuncFormatter(lambda v, _: f"{v:.0f}%".replace("-", "−"))
COL = {"Headline": S[0], "Full short leg": S[1], "Long leg only": S[2], ALT: S[3], "Long leg, short PGJ": S[4]}
NAME = {"Headline": "Optimized borrow random forest" if MODEL == "RF" else "\\$10m short leg (headline)", "Full short leg": "Full short leg, retail borrow", "Long leg only": "Long leg only", "Random forest": "Random-forest forecasts", "Ensemble": "Ensemble forecasts", "Long leg, short PGJ": "Long leg, short PGJ"}; dn = lambda k: NAME.get(k, k)
UD = pd.read_csv(f"{R}/{TAG}_unit_daily.csv", index_col=0, parse_dates=True); V60 = UD["vol60"] * 100; VB = UD["vol"] * 100
def load(tag):
    return dict(d=pd.read_csv(f"{R}/{tag}_daily.csv", index_col=0, parse_dates=True), m=pd.read_csv(f"{R}/{tag}_monthly.csv", index_col=0, parse_dates=True), E=pd.read_csv(f"{R}/{tag}_equity_curves.csv", index_col=0, parse_dates=True),
                V=pd.read_csv(f"{R}/{tag}_variants.csv", index_col=0), mc=json.load(open(f"{R}/{tag}_mc.json")), B=pd.read_csv(f"{R}/{tag}_bootstrap.csv"), fan=pd.read_csv(f"{R}/{tag}_bootstrap_fan.csv"),
                perm=pd.read_csv(f"{R}/{tag}_null.csv")["null_sharpe"].values)
FULL = load(TAG); FIVE = load(f"{TAG}_5y")

# ---------- fig2: equity and drawdown, full window ----------
E = FULL["E"]; fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), dpi=200, sharex=True, gridspec_kw={"height_ratios": [2.4, 1]})
ax = axes[0]; style(ax)
for k in ["Long leg only", "Full short leg", "Long leg, short PGJ", ALT, "Headline"]:
    ax.plot(E.index, E[k].values, color=COL[k], linewidth=2.2 if k == "Headline" else 1.5, solid_joinstyle="round", label=dn(k))
ends = E.iloc[-1].sort_values(); ypos = ends.copy(); gap = ends.max() * 0.045
for i in range(1, len(ypos)):
    if ypos.iloc[i] - ypos.iloc[i - 1] < gap: ypos.iloc[i] = ypos.iloc[i - 1] + gap
for k in ends.index: ax.text(E.index[-1] + pd.offsets.Day(60), ypos[k], f"{dn(k)}  \\${ends[k]/1e6:,.2f}m", fontsize=8.5, color=INK2, va="center")
ax.axhline(500000, color=AXIS, linewidth=1); ax.yaxis.set_major_formatter(kusd); ax.set_xlim(E.index[0], E.index[-1] + pd.offsets.Day(1500)); ax.set_xticks(pd.to_datetime([f"{y}-01-01" for y in range(2010, 2027, 2)])); ax.set_xticklabels(range(2010, 2027, 2))
ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3); ax.set_ylim(None, ends.max() * 1.18)
title(ax, "$500,000 account, 10% volatility target, January 2010 to August 2026 (after commissions, borrow and cash interest)", 11.5)
ax = axes[1]; style(ax); eq = E["Headline"]; dd = (eq / eq.cummax() - 1) * 100
ax.fill_between(dd.index, dd.values, 0, color=S[0], alpha=0.25, linewidth=0); ax.plot(dd.index, dd.values, color=S[0], linewidth=1.2)
ax.yaxis.set_major_formatter(pct); t0 = dd.idxmin(); ax.set_ylim(dd.min() * 1.12, 0.5); ax.annotate(f"−{-dd.min():.1f}%, {t0:%b %Y}", (t0, dd.min()), xytext=(10, 2), textcoords="offset points", fontsize=8.5, color=INK2, va="center")
title(ax, f"{dn('Headline')}: drawdown from the previous equity peak", 10.5)
fig.subplots_adjust(left=0.065, right=0.985, top=0.94, bottom=0.05, hspace=0.16); fig.savefig(f"{R}/fig2_bt500k{SUF}_equity.png"); plt.close(fig); print("fig2 written")

# ---------- signal, volatility, exposure, margin (optionally with an equity panel on top) ----------
def fig_account(X, out, equity=None, pdf=None):
    d, m = X["d"], X["m"]; w0, w1 = d.index[0], d.index[-1]; v60 = V60.loc[w0:w1]; vb = VB.loc[w0:w1]
    n = 5 if equity else 4; fig, axes = plt.subplots(n, 1, figsize=(12, 2.9 * n), dpi=200, sharex=True, gridspec_kw={"height_ratios": ([1.35] if equity else []) + [1.15, 1, 1, 1]}); axes = list(axes)
    if equity:
        ax = axes.pop(0); style(ax); E = X["E"]
        for k in ["Long leg only", "Full short leg", ALT, "Headline"]: ax.plot(E.index, E[k].values, color=COL[k], linewidth=2.2 if k == "Headline" else 1.5, solid_joinstyle="round", label=f"{dn(k)}: \\${E[k].iloc[-1]/1000:,.0f}k")
        ax.axhline(500000, color=AXIS, linewidth=1); ax.yaxis.set_major_formatter(kusd); ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=4, title="final equity", title_fontsize=8, alignment="left"); title(ax, equity, 11.5)
        ax.set_ylim(None, E.max().max() * 1.14)
    ax = axes[0]; style(ax); g = m["gap"] * 100; nl = m["n_long"]; ns = m["n_short"]
    ax.bar(m.index, g.values, width=24, color=S[0], alpha=0.8, linewidth=0, label="Forecast gap: mean forecast of the long leg minus the short leg (% next month)")
    ax2 = ax.twinx(); ax2.plot(m.index, ns.values, color=S[1], linewidth=1.6, drawstyle="steps-post", label="Names in the short leg (right axis)"); ax2.plot(m.index, nl.values, color=MUTED, linewidth=1.2, drawstyle="steps-post", label="Names in the long leg (right axis)")
    ax2.set_ylim(0, max(nl.max(), ns.max()) * 1.9); ax2.tick_params(length=0, colors=MUTED); ax2.spines[["top", "left", "right"]].set_visible(False); ax2.set_ylabel("Names", fontsize=9)
    ax.set_ylabel("% per month", fontsize=9); ax.set_ylim(0, g.max() * 1.75); h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="upper left", ncol=3)
    title(ax, "Signal: the forecast spread the model expects between the legs, and the number of names held", 11 if equity else 11.5)
    ax = axes[1]; style(ax)
    ax.plot(v60.index, v60.values, color=MUTED, linewidth=1.1, label="60-day realised volatility of the unit long-short (1x a side)"); ax.plot(vb.index, vb.values, color=S[0], linewidth=2, solid_joinstyle="round", label="Sizing volatility (70% 60-day + 30% long-run mean)")
    ax.set_ylabel("% per year", fontsize=9); ax.set_ylim(0, v60.max() * 1.25); ax.yaxis.set_major_formatter(pct); ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
    title(ax, "Volatility of the long-short portfolio (annualised)")
    ax = axes[2]; style(ax); lv = d["long_value"]; sv = d["short_value"]
    ax.fill_between(lv.index, lv.values, 0, color=S[0], alpha=0.85, linewidth=0, label="Long leg, market value"); ax.fill_between(sv.index, -sv.values, 0, color=S[1], alpha=0.85, linewidth=0, label="Short leg, market value")
    ax.axhline(0, color=AXIS, linewidth=1); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${abs(v)/1000:,.0f}k")); ax.set_ylabel("Dollars", fontsize=9); ax.set_ylim(-sv.max() * 1.15, lv.max() * 1.45); ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
    e = m[m.exposure > 0]["exposure"]; title(ax, f"Position: dollars a side (10% volatility target on current equity, capped at 80% a side; average {e.mean()*100:.0f}% of equity a side, range {e.min()*100:.0f} to {e.max()*100:.0f}%)")
    ax = axes[3]; style(ax); mp = d["margin_pct"] * 100; ms = d["margin_stress_pct"] * 100; mt = d["maint_pct"] * 100
    ax.plot(ms.index, ms.values, color=INK2, linewidth=1.1, linestyle=(0, (4, 2)), label="If the broker required 75% a side (house margin on volatile ADRs)")
    ax.plot(mp.index, mp.values, color=S[0], linewidth=2, solid_joinstyle="round", label="Reg T initial margin, 50% a side"); ax.plot(mt.index, mt.values, color=S[2], linewidth=1.2, label="Reg T maintenance, 25% long and 30% short")
    ax.axhline(100, color=S[1], linewidth=1, linestyle=(0, (2, 2))); ax.text(w0 + (w1 - w0) * 0.995, 101.5, "100% = all equity pledged", fontsize=8, color=S[1], ha="right", va="bottom")
    ax.set_ylim(0, max(ms.max(), mp.max(), 110) * 1.3); ax.yaxis.set_major_formatter(pct); ax.set_ylabel("Share of account equity", fontsize=9); ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=1)
    title(ax, f"Margin deployed: average {mp.mean():.0f}% of equity at Reg T initial, peak {mp.max():.0f}%; maintenance peak {mt.max():.0f}%")
    fig.subplots_adjust(left=0.07, right=0.95, top=0.96 if not equity else 0.965, bottom=0.035, hspace=0.2 if not equity else 0.24); fig.savefig(f"{R}/{out}"); pdf is not None and pdf.savefig(fig); plt.close(fig); print(out, "written")

# ---------- Monte Carlo: Sharpe densities, equity fan, drawdown, no-skill null ----------
def fig_mc(X, out, strategies, pdf=None):
    B, fan, perm, mc, d, m = X["B"], X["fan"], X["perm"], X["mc"], X["d"], X["m"]; T = len(fan); cap = 500000
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=200)
    ax = axes[0, 0]; style(ax); allv = np.concatenate([B[f"{k}|sharpe"].values for k in strategies]); bins = np.linspace(np.floor(np.percentile(allv, 0.05) * 2) / 2, np.ceil(np.percentile(allv, 99.95) * 2) / 2, 61); top = 0
    for k in strategies:
        h, e = np.histogram(B[f"{k}|sharpe"], bins=bins, density=True); c = (e[:-1] + e[1:]) / 2; med = B[f"{k}|sharpe"].median(); top = max(top, h.max())
        ax.plot(c, h, color=COL[k], linewidth=2, label=f"{dn(k)}: median " + f"{med:+.2f}".replace("-", "−"), solid_joinstyle="round"); ax.plot([med, med], [0, np.interp(med, c, h)], color=COL[k], linewidth=1.4, linestyle=(0, (3, 2)))
    ax.plot([0, 0], [0, top * 1.04], color=AXIS, linewidth=1, zorder=1); ax.set_xlabel(f"Annualised Sharpe of a bootstrapped {T}-month path", fontsize=9.5); ax.set_ylabel("Density", fontsize=9.5); ax.set_ylim(None, top * 1.38)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2, title="dashed line = median of each distribution", title_fontsize=8, alignment="left")
    bs = mc["bootstrap_sharpe"]["Headline"]; title(ax, f"Sharpe: {dn('Headline').lower() if MODEL == 'RF' else 'headline'} median {bs['p50']:+.2f}, 90% interval {bs['p05']:+.2f} to {bs['p95']:+.2f}")
    ax = axes[0, 1]; style(ax); x = np.arange(1, T + 1) / 12; act = m["equity_start"].shift(-1).fillna(d["equity"].iloc[-1]).values
    ax.fill_between(x, fan["p05"], fan["p95"], color=BAND[0], linewidth=0, label="5th to 95th percentile"); ax.fill_between(x, fan["p25"], fan["p75"], color=BAND[1], linewidth=0, label="25th to 75th percentile")
    ax.plot(x, fan["p50"], color=S[0], linewidth=2, label="Median path"); ax.plot(x, act, color=INK, linewidth=1.4, label="Actual backtest path")
    lo, hi = min(fan["p05"].min(), act.min()) * 0.8, max(fan["p95"].max(), act.max()) * 1.25; ticks = [t for t in [250000, 500000, 1e6, 2e6, 4e6, 8e6] if lo <= t <= hi]
    ax.axhline(cap, color=AXIS, linewidth=1); ax.set_yscale("log"); ax.set_ylim(lo, hi); ax.set_yticks(ticks); ax.yaxis.set_major_formatter(kusd); ax.minorticks_off()
    ax.set_xlabel("Years from start", fontsize=9.5); ax.set_ylabel("Account equity (log scale)", fontsize=9.5); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    bp = mc["bootstrap_pnl"]["Headline"]; title(ax, f"P&L: final equity \\${bp['final_p05']/1e6:,.2f}m to \\${bp['final_p95']/1e6:,.2f}m, median \\${bp['final_p50']/1e6:,.2f}m")
    ax = axes[1, 0]; style(ax); dd = B["Headline|maxdd_pct"]; e = d["equity"].resample("ME").last(); obs = ((e / np.maximum(e.cummax(), cap) - 1).min()) * 100
    ax.hist(dd, bins=np.linspace(min(-30, np.floor(np.percentile(dd, 0.05) / 10) * 10), 0, 49), color=S[0], alpha=0.35, edgecolor=SURF, linewidth=1); ax.axvline(obs, color=INK, linewidth=1.6); ax.set_ylim(None, ax.get_ylim()[1] * 1.12)
    ax.text(obs - 0.015 * ax.get_xlim()[0], ax.get_ylim()[1] * 0.97, f"backtest −{-obs:.1f}%\n(month-end)", fontsize=8.5, color=INK2, va="top", ha="left"); ax.xaxis.set_major_formatter(pct)
    ax.set_xlabel("Worst peak-to-trough drawdown on a bootstrapped path (month-end equity)", fontsize=9.5); ax.set_ylabel("Count of paths", fontsize=9.5)
    title(ax, f"Drawdown: median −{-bp['maxdd_p50']:.0f}%, 1 path in 20 worse than −{-bp['maxdd_p05']:.0f}%")
    ax = axes[1, 1]; style(ax); nl = mc["nulls"]; lim = max(1.2, np.ceil((max(np.abs(perm).max(), nl["observed"]) + 0.1) * 5) / 5)
    ax.hist(perm, bins=np.linspace(-lim, lim, 49), color=S[0], alpha=0.35, edgecolor=SURF, linewidth=1); ax.axvline(nl["observed"], color=INK, linewidth=1.6); ax.set_ylim(None, ax.get_ylim()[1] * 1.22)
    ax.text(nl["observed"] - 0.05 * lim, ax.get_ylim()[1] * 0.97, f"observed {nl['observed']:+.2f}\n{int(round(nl['p_null'] * len(perm)))} of {len(perm):,} random sorts as high", fontsize=8.5, color=INK2, va="top", ha="right")
    ax.set_xlabel("Annualised Sharpe with the names picked at random (same universe, sizing, costs, borrow)", fontsize=9.5); ax.set_ylabel("Count of random sorts", fontsize=9.5)
    title(ax, f"Is the selection luck? Random-sort null for the {dn('Headline').lower() if MODEL == 'RF' else 'headline'}")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.95, bottom=0.07, wspace=0.2, hspace=0.3); fig.savefig(f"{R}/{out}"); pdf is not None and pdf.savefig(fig); plt.close(fig); print(out, "written")

fig_account(FULL, f"fig3_bt500k{SUF}_signal_vol_margin.png")
fig_mc(FULL, f"fig4_bt500k{SUF}_montecarlo.png", ["Headline", ALT, "Full short leg", "Long leg only"])
with PdfPages(f"{R}/{TAG}_5y_pages.pdf") as pdf:
    fig_account(FIVE, f"fig5_bt500k{SUF}_5y_account.png", equity="Last five years: a new $500,000 account, September 2021 to August 2026 (10% volatility target, after costs)", pdf=pdf)
    fig_mc(FIVE, f"fig6_bt500k{SUF}_5y_montecarlo.png", ["Headline", ALT, "Full short leg", "Long leg only"], pdf=pdf)
