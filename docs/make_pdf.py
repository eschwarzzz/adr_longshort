"""Assemble docs/MODEL.md, docs/STATS.md, the figures in results/ and the model tables into one LaTeX document and PDF (pandoc + tectonic / XeTeX).

    python3 docs/make_pdf.py

Output: docs/CHINA_ADR_RF.tex and docs/CHINA_ADR_RF.pdf."""
import os, re, subprocess, datetime, pandas as pd
DOCS = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(DOCS); R = f"{ROOT}/results"; STEM = "CHINA_ADR_RF"
def body(path):
    """Drop the file's title line, push its sections one level down."""
    lines = open(path).read().splitlines(); lines = [l for l in lines if not l.startswith("# ")]
    return "\n".join(("#" + l) if l.startswith("## ") else l for l in lines).strip() + "\n"
model = body(f"{DOCS}/MODEL.md"); stats = body(f"{DOCS}/STATS.md")
# the PDF carries no code or file references: targeted rewrites, then every remaining "File(s): ..." line and backticked name is removed
PDF_REWRITES = [
    (" `results/predictions_rf.csv` lists every ticker-month the model has scored.", ""),
    ("Daily prices are cleaned first (`src/clean_prices.py`): ", "Daily prices are cleaned first: "),
    ("from that name's own history up to t (`src/features.py`):", "from that name's own history up to t:"),
    ("The model sees only these sixteen ranks, in the order given in `model/feature_spec.json`.", "The model sees only these sixteen ranks."),
    ("scikit-learn `RandomForestRegressor(n_estimators=300, max_features=0.5, random_state=0)`, with `max_depth` ∈ {2, 4, 6} and `min_samples_leaf` ∈ {20, 50} chosen by validation mean squared error.",
     "A random forest of 300 trees, each split drawing on half of the characteristics, with maximum tree depth in {2, 4, 6} and minimum leaf size in {20, 50} chosen by validation mean squared error."),
    ("The hyperparameters chosen each year are in `model/rf_hyperparams_by_year.csv`:", "The hyperparameters chosen each year are listed in Appendix A:"),
    ("Impurity-based importance for `rf_2026` (`model/feature_importance.csv`):", "Impurity-based importance for the 2026 forest (Appendix B):"),
    ("### From forecasts to positions (`src/portfolio.py`)", "### From forecasts to positions"),
    (" Generated from the files in `results/` by `docs/make_stats.py`.", ""),
]
for a, b in PDF_REWRITES:
    assert (model + stats).count(a) == 1, a[:50]
    model = model.replace(a, b); stats = stats.replace(a, b)
stats = re.sub(r"(?m) ?Files?: .*$", "", stats); stats = re.sub(r"\n{3,}", "\n\n", stats)
for txt in (model, stats): assert "`" not in txt, re.findall(r"`[^`]*`", txt)
FIG = {"4. The $500,000 account": [("results/fig2_bt500k_rf_equity.png", "Figure 2. Account equity and drawdown, January 2010 to August 2026: the optimized-borrow random forest against the same account on the ensemble forecasts, the full short leg at retail borrow, the long leg alone and the long leg hedged with PGJ.", "width=100%"),
                               ],
       "5. Monte Carlo": [("results/fig3_bt500k_rf_signal_vol_margin.png", "Figure 3. Signal (forecast gap and names held), volatility of the long-short, dollars a side, and margin deployed, 2010 to 2026.", "height=88%"),
                          ("results/fig4_bt500k_rf_montecarlo.png", "Figure 4. Bootstrap Sharpe densities, equity fan against the actual path, drawdown distribution, and the random-sort null.", "width=100%")],
       "6. The last five years": [("results/fig5_bt500k_rf_5y_account.png", "Figure 5. A fresh account from September 2021: equity, signal, volatility, dollars a side, margin.", "height=88%"),
                              ("results/fig6_bt500k_rf_5y_montecarlo.png", "Figure 6. Monte Carlo on the 60 months.", "width=100%")]}
# insert each figure block at the end of its section (before the next ### heading)
parts = re.split(r"(?m)^(?=### )", stats); out = []
fig = lambda f: f"![{f[1]}]({f[0]}){{{f[2]}}}"
for p in parts:
    for key, figs in FIG.items():
        if p.startswith("### " + key):
            if key.startswith("4."):                      # the equity chart goes under the account table, the tall chart closes the section
                p = p.replace("\nCalendar years, return on the account", "\n" + fig(figs[0]) + "\n\nCalendar years, return on the account"); figs = figs[1:]
            if key.startswith("7."): pass
            if key.startswith(("5.", "6.")): p = p.splitlines()[0] + "\n"             # graphs only
            p = p.rstrip("\n") + "\n\n" + "\n\n".join(fig(f) for f in figs) + "\n\n"
    out.append(p)
stats = "".join(out)
H = pd.read_csv(f"{ROOT}/model/rf_hyperparams_by_year.csv", index_col=0); I = pd.read_csv(f"{ROOT}/model/feature_importance.csv", index_col=0)
hyp = "| Test year | Training rows (names) | Validation rows | max_depth | min_samples_leaf | Validation MSE |\n|---|---|---|---|---|---|\n" + "\n".join(f"| {y} | {int(r.train_rows):,} ({int(r.train_names)}) | {int(r.val_rows):,} | {int(r.max_depth)} | {int(r.min_samples_leaf)} | {r.val_mse:.4f} |" for y, r in H.iterrows())
imp = "| Characteristic | rf_2026 | rf_next |\n|---|---|---|\n" + "\n".join(f"| {f} | {r.rf_2026:.3f} | {r.rf_next:.3f} |" for f, r in I.iterrows())
allf = sorted(os.path.relpath(os.path.join(d, f), ROOT) for d, _, fs in os.walk(ROOT) for f in fs if not f.startswith((".", "_")) and "__pycache__" not in d and not f.endswith((".tex", ".pdf")))
groups = {}
for p in allf: groups.setdefault(os.path.dirname(p) or "top level", []).append(os.path.basename(p))
files = "\n\n".join((f"**{g}/**: " if g != "top level" else "**Top level**: ") + ", ".join(f"`{f}`" for f in fs) if g != "results" else "**results/**:\n\n\\begin{multicols}{2}\\begin{itemize}\\tightlist\n" + "\n".join("\\item \\texttt{" + f.replace("_", "\\_") + "}" for f in fs) + "\n\\end{itemize}\\end{multicols}" for g, fs in groups.items())
doc = f"""# Chinese ADR random-forest long-short

Part I describes the model and the rule that turns its forecasts into positions; Part II gives the backtest statistics; the appendix lists the hyperparameters chosen each year and the feature importances.

## Part I. Model description

{model}
## Part II. Backtest statistics

{stats}
## Appendix

### A. Hyperparameters chosen by validation, by test year

Each December before test year Y the forest is trained on 2001..Y−3 and the six candidates (max_depth 2, 4, 6 × min_samples_leaf 20, 50) are scored on Y−2..Y−1. Refitting with these settings reproduces the saved forecasts to machine precision.

{hyp}

### B. Impurity-based feature importance

{imp}
"""
# characters the body font does not carry go to math; the dollar sign is plain text throughout (tex_math_dollars is off)
for a, b in [("σ", r"\(\sigma\)"), ("∈", r"\(\in\)"), ("≥", r"\(\geq\)"), ("≤", r"\(\leq\)"), ("×", r"\(\times\)"), ("R²", r"R\(^2\)"), ("→", r"\(\rightarrow\)"), ("÷", r"\(\div\)")]: doc = doc.replace(a, b)
def fix_table(lines):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]; n = max(len(r) for r in rows)
    w = [max(len(r[j]) if j < len(r) else 0 for i, r in enumerate(rows) if i != 1) for j in range(n)]; w = [max(3, min(x, 110)) for x in w]
    lines[1] = "|" + "|".join("-" * x for x in w) + "|"; return lines
out, buf = [], []
for l in doc.splitlines() + [""]:
    if l.startswith("|"): buf.append(l)
    else:
        if buf: out += fix_table(buf) if len(buf) > 1 and set(buf[1].replace("|", "").strip()) <= set("-: ") else buf; buf = []
        out.append(l)
tmp = f"{R}/_{STEM}_latex.md"; open(tmp, "w").write("\n".join(out))
HEAD = r"""\usepackage{etoolbox}\AtBeginEnvironment{longtable}{\footnotesize}
\usepackage{float}\floatplacement{figure}{H}
\setlength{\parskip}{5pt plus 1pt}\setlength{\parindent}{0pt}
\usepackage{caption}\captionsetup{font=small,skip=4pt,labelformat=empty}
\renewcommand{\arraystretch}{1.15}\usepackage{multicol}
\usepackage{unicode-math}"""
HEAD = HEAD.replace("\n\\usepackage{unicode-math}", "")
open(f"{R}/_pdf_header.tex", "w").write(HEAD)
common = ["--from", "markdown+pipe_tables-tex_math_dollars+tex_math_single_backslash", "--shift-heading-level-by=-1", "--columns=60", f"--resource-path={ROOT}", "-V", "geometry:margin=2.1cm", "-V", "papersize=a4", "-V", "fontsize=10pt",
          "-V", "mainfont=texgyreheros-regular.otf", "-V", "mainfontoptions=BoldFont=texgyreheros-bold.otf,ItalicFont=texgyreheros-italic.otf,BoldItalicFont=texgyreheros-bolditalic.otf",
          "-V", "monofont=texgyrecursor-regular.otf", "-V", "monofontoptions=BoldFont=texgyrecursor-bold.otf,Scale=0.9", "-V", "colorlinks=true", "-V", "linkcolor=blue", "-V", "urlcolor=blue",
          "-V", "date=" + datetime.date.today().strftime("%-d %B %Y"), f"--include-in-header={R}/_pdf_header.tex"]
subprocess.run(["pandoc", tmp, "-s", "-o", f"{DOCS}/{STEM}.tex"] + common, check=True)
subprocess.run(["pandoc", tmp, "--pdf-engine=tectonic", "-o", f"{DOCS}/{STEM}.pdf"] + common, check=True, cwd=ROOT)
for f in (tmp, f"{R}/_pdf_header.tex"): os.remove(f)
print(f"docs/{STEM}.tex and docs/{STEM}.pdf written")
