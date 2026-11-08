"""Backtest de la règle d'entrée « RSI < seuil » (sans Streamlit).

Convention anti-anticipation : le signal est calculé avec les données jusqu'à la CLÔTURE du jour J ;
l'entrée se fait à l'OUVERTURE du jour J+1. Les sorties fondées sur la clôture (RSI > 50, durée) sont
elles aussi exécutées à l'ouverture du jour suivant. Le Stop-Loss (prix d'entrée − k × ATR, ATR connu au
jour J) est exécuté en séance : au niveau du stop, ou à l'ouverture en cas de gap sous le stop.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core import ATR_MULT, OBV_LOOKBACK, RSI_THRESHOLD, add_advanced_columns, rsi

VARIANTS = {
    "RSI seul": "none",
    "RSI + OBV ou MACD": "either",
    "RSI + OBV et MACD": "both",
}
MIN_TRADES = 30        # en dessous, les statistiques ne sont pas fiables
TRAIN_SHARE = 0.70


@dataclass
class Params:
    rsi_entry: float = RSI_THRESHOLD
    rsi_exit: float | None = 50.0     # None = sortie RSI désactivée
    atr_mult: float | None = ATR_MULT  # None = Stop-Loss désactivé
    hold_days: int | None = 20         # None = durée fixe désactivée
    fee: float = 0.001                 # par transaction (achat et vente)


def prepare(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Indicateurs causaux (la valeur du jour J n'utilise que les données jusqu'à J)."""
    d = add_advanced_columns(ohlcv)
    d["RSI"] = rsi(d["Close"])
    d["OBV_ok"] = ((d["OBV"] - d["OBV"].shift(OBV_LOOKBACK)) >= 0)       # = OBV 🟢 ou 🟡 (obv_signal)
    h, p = d["MACD_Hist"], d["MACD_Hist"].shift(1)
    d["MACD_ok"] = pd.Series(np.where(h >= 0, h >= p, h > p), index=d.index) & p.notna()   # = MACD 🟢 ou 🟡
    return d


def entry_signal(d: pd.DataFrame, variant: str, rsi_entry: float) -> pd.Series:
    """Signal calculé à la clôture de chaque jour (True = on voudrait entrer le lendemain)."""
    base = (d["RSI"] < rsi_entry) & d["ATR"].notna()
    mode = VARIANTS[variant]
    if mode == "either":
        return base & (d["OBV_ok"] | d["MACD_ok"])
    if mode == "both":
        return base & d["OBV_ok"] & d["MACD_ok"]
    return base


def simulate(d: pd.DataFrame, variant: str, params: Params, start: int = 0, end: int | None = None):
    """Simule sur les lignes [start, end). Retourne (trades, equity) ; equity est indexée par date, départ à 1."""
    end = len(d) if end is None else end
    sig = entry_signal(d, variant, params.rsi_entry).to_numpy()
    o, h, l, c = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    r, atr = d["RSI"].to_numpy(float), d["ATR"].to_numpy(float)
    fee = params.fee
    trades, i = [], start
    while i < end - 1:
        if not sig[i]:
            i += 1
            continue
        e = i + 1                                   # entrée à l'ouverture du jour suivant le signal
        entry = o[e]
        if not entry > 0:
            i += 1
            continue
        stop = entry - params.atr_mult * atr[i] if params.atr_mult else None
        exit_idx = exit_px = None
        reason = "fin de période"
        for j in range(e, end):
            if stop is not None and l[j] <= stop:
                exit_idx, exit_px, reason = j, min(o[j], stop), "stop ATR"
                break
            if j + 1 >= end:
                break
            if params.rsi_exit is not None and r[j] > params.rsi_exit:
                exit_idx, exit_px, reason = j + 1, o[j + 1], "RSI"
                break
            if params.hold_days is not None and j + 1 - e >= params.hold_days:
                exit_idx, exit_px, reason = j + 1, o[j + 1], "durée"
                break
        if exit_idx is None:
            exit_idx, exit_px = end - 1, c[end - 1]
        ret = exit_px / entry * (1 - fee) / (1 + fee) - 1
        trades.append({"signal": d.index[i], "entry_date": d.index[e], "entry": entry,
                       "exit_date": d.index[exit_idx], "exit": exit_px, "ret": ret,
                       "days": exit_idx - e, "reason": reason, "ei": e, "xi": exit_idx})
        i = exit_idx                                # flat dès la sortie ; nouveau signal à la clôture de ce jour
    tr = pd.DataFrame(trades, columns=["signal", "entry_date", "entry", "exit_date", "exit", "ret", "days",
                                       "reason", "ei", "xi"])
    return tr.drop(columns=["ei", "xi"]), _equity(d, tr, c, start, end, fee)


def _equity(d, tr, close, start, end, fee) -> pd.Series:
    eq = np.ones(end - start)
    cap = 1.0
    last = start
    for t in tr.itertuples():
        eq[last - start:t.ei - start] = cap
        units = cap / (t.entry * (1 + fee))
        eq[t.ei - start:t.xi - start] = units * close[t.ei:t.xi]
        cap = units * t.exit * (1 - fee)
        eq[t.xi - start] = cap
        last = t.xi + 1
    eq[last - start:] = cap
    return pd.Series(eq, index=d.index[start:end])


def buy_and_hold(d: pd.DataFrame, start: int, end: int, fee: float) -> pd.Series:
    """Achat à l'ouverture du lendemain du premier jour (comme la stratégie), frais d'entrée et de sortie inclus."""
    e = start + 1
    base = d["Open"].iloc[e] * (1 + fee)
    eq = pd.Series(1.0, index=d.index[start:end])
    eq.iloc[1:] = d["Close"].iloc[e:end].to_numpy() * (1 - fee) / base
    return eq


def max_drawdown(eq: pd.Series) -> float:
    return float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0


def segments(n: int, train_share: float = TRAIN_SHARE) -> dict:
    cut = int(n * train_share)
    return {"Construction (70 %)": (0, cut), "Hors échantillon (30 %)": (cut, n)}


def summarize(trades: pd.DataFrame, equity: pd.Series, bh: pd.Series, n_assets: int = 1) -> dict:
    rets = trades["ret"] if len(trades) else pd.Series(dtype=float)
    return {
        "Trades": int(len(trades)),
        "Réussite": float((rets > 0).mean()) if len(rets) else np.nan,
        "Gain moyen": float(rets.mean()) if len(rets) else np.nan,
        "Pire perte": float(rets.min()) if len(rets) else np.nan,
        "Temps investi": float(trades["days"].sum() / (len(equity) * n_assets)) if len(equity) else np.nan,
        "Max drawdown": max_drawdown(equity),
        "Rendement total": float(equity.iloc[-1] - 1) if len(equity) else np.nan,
        "Buy & hold": float(bh.iloc[-1] - 1) if len(bh) else np.nan,
        "B&H drawdown": max_drawdown(bh),
    }


def run_asset(ohlcv: pd.DataFrame, params: Params) -> dict:
    """{segment: {variante: {"trades", "equity", "bh", "stats"}}} pour un actif."""
    d = prepare(ohlcv)
    out = {}
    for seg, (a, b) in segments(len(d)).items():
        if b - a < 30:
            continue
        bh = buy_and_hold(d, a, b, params.fee)
        out[seg] = {}
        for v in VARIANTS:
            tr, eq = simulate(d, v, params, a, b)
            out[seg][v] = {"trades": tr, "equity": eq, "bh": bh, "stats": summarize(tr, eq, bh)}
    return out


def aggregate(results: dict) -> dict:
    """Agrège plusieurs actifs ({actif: run_asset}) : trades poolés, courbes en moyenne équipondérée."""
    agg = {}
    for asset, res in results.items():
        for seg, variants in res.items():
            for v, r in variants.items():
                slot = agg.setdefault(seg, {}).setdefault(v, {"trades": [], "equity": [], "bh": []})
                slot["trades"].append(r["trades"].assign(asset=asset))
                slot["equity"].append(r["equity"])
                slot["bh"].append(r["bh"])
    out = {}
    for seg, variants in agg.items():
        out[seg] = {}
        for v, s in variants.items():
            trades = pd.concat(s["trades"], ignore_index=True)
            eq, bh = _mean_curve(s["equity"]), _mean_curve(s["bh"])
            out[seg][v] = {"trades": trades, "equity": eq, "bh": bh,
                           "stats": summarize(trades, eq, bh, len(s["equity"]))}
    return out


def _mean_curve(curves: list) -> pd.Series:
    idx = curves[0].index
    for cv in curves[1:]:
        idx = idx.union(cv.index)
    aligned = [cv.reindex(idx).ffill().fillna(1.0) for cv in curves]
    return pd.concat(aligned, axis=1).mean(axis=1)


def welch_t(a: pd.Series, b: pd.Series) -> float:
    """t de Welch (différence de moyennes) ; nan si pas calculable."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def verdict(train_trades: pd.DataFrame, test_trades: pd.DataFrame, baseline_test: pd.DataFrame | None = None) -> str:
    """Lecture honnête : peu de trades, résultats instables ou indistinguables du hasard."""
    n_tr, n_te = len(train_trades), len(test_trades)
    if n_te < MIN_TRADES or n_tr < MIN_TRADES:
        return f"⚪ Non concluant : trop peu de trades ({n_tr} / {n_te}, minimum {MIN_TRADES})"
    r_tr, r_te = train_trades["ret"], test_trades["ret"]
    t_te = r_te.mean() / (r_te.std(ddof=1) / np.sqrt(n_te)) if r_te.std(ddof=1) > 0 else float("nan")
    if np.sign(r_tr.mean()) != np.sign(r_te.mean()):
        return "🟠 Instable : le gain moyen change de signe entre construction et test"
    if not abs(t_te) >= 2:
        return f"⚪ Indistinguable du hasard : gain moyen hors échantillon non significatif (t = {t_te:.1f})"
    if baseline_test is not None and len(baseline_test) >= MIN_TRADES:
        t_diff = welch_t(r_te, baseline_test["ret"])
        if not abs(t_diff) >= 2:
            return f"⚪ Pas mieux que « RSI seul » (écart non significatif, t = {t_diff:.1f})"
    return ("🟢 Résultat hors échantillon significatif (à relativiser : passé ≠ futur)"
            if r_te.mean() > 0 else "🔴 Perdant de façon significative hors échantillon")
