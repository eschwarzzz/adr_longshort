# docs

- `MODEL.md`: the model and the rule that turns its forecasts into positions. Hand-written; the numbers quoted in it come from `results/` and `model/`.
- `STATS.md`: the backtest statistics, generated from `results/` and `model/` by `make_stats.py`. Rerun after `backtest.py`.
- `CHINA_ADR_RF.pdf` (and `.tex`): both documents, the figures and the model tables as one LaTeX document, built by `make_pdf.py` (needs pandoc and tectonic). The PDF carries no file or code references; the markdown files do.
