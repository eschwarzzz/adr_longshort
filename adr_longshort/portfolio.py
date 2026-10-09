"""From forecasts to positions: the quintile rule with the $3m short-leg floor, and the volatility-targeted sizing used in the $500k backtest."""
import numpy as np, pandas as pd
LIQ = 1e6; SHORT_FLOOR = 3e6; MIN_N = 25; MIN_SHORT_POOL = 10; MIN_SHORT = 5
TARGET = 0.10; CAP = 0.80; PRIOR_VOL = 0.40

def holdings(month, col="p_RF", liq=LIQ, short_floor=SHORT_FLOOR, min_n=MIN_N):
    """month: the panel rows of one month-end with columns ticker, adv, tradeable (optional) and the forecast column.
    Returns dict(long=[tickers], short=[tickers], ...) or None when the month is flat (too few names)."""
    g = month[month.adv >= liq]
    if "tradeable" in g: g = g[g.tradeable]
    if len(g) < min_n: return None
    g = g.sort_values(col); k = len(g) // 5; long = g.tail(k)                                   # top quintile of the liquid universe, equal weight
    gs = g[g.adv >= short_floor] if short_floor else g
    if len(gs) < MIN_SHORT_POOL: return None
    short = gs.head(max(len(gs) // 5, MIN_SHORT))                                              # bottom quintile among names above the floor, equal weight, at least five
    return dict(long=list(long.ticker), short=list(short.ticker), adv_short=short.adv.values, n_liquid=len(g), n_short_pool=len(gs), forecast_gap=float(long[col].mean() - short[col].mean()))

def sizing_vol(unit_ls_daily):
    """unit_ls_daily: daily returns of the one-dollar-a-side long-short. Sizing volatility = 0.7 x 60-day realised + 0.3 x its expanding mean, annualised."""
    v60 = unit_ls_daily.rolling(60, min_periods=20).std() * np.sqrt(252); return 0.7 * v60 + 0.3 * v60.expanding(20).mean()

def dollars_a_side(equity, sigma, target=TARGET, cap=CAP):
    """Exposure per leg: min(cap, target / sigma) x equity. sigma from sizing_vol at the month-end, PRIOR_VOL if none is available yet."""
    sigma = PRIOR_VOL if sigma is None or np.isnan(sigma) else sigma; return min(cap, target / sigma) * equity

def weights(h, dollars):
    """Equal-weight dollar positions: +dollars/n for each long, -dollars/n for each short."""
    w = {t: dollars / len(h["long"]) for t in h["long"]}; w.update({t: -dollars / len(h["short"]) for t in h["short"]}); return w
