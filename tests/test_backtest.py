import numpy as np
import pandas as pd
import pytest

import core
from backtest import Params, entry_signal, prepare, run_asset, aggregate, simulate, verdict


def make_ohlcv(n=900, seed=1, drift=0.0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.02, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    vol = rng.integers(1_000, 5_000, n).astype(float)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


P = Params(rsi_entry=35, rsi_exit=50, atr_mult=2.0, hold_days=20, fee=0.0)


def test_entry_happens_at_next_open_not_signal_close():
    df = make_ohlcv()
    d = prepare(df)
    trades, _ = simulate(d, "RSI seul", P)
    assert len(trades) > 5
    for t in trades.itertuples():
        i = d.index.get_loc(t.signal)
        assert t.entry_date == d.index[i + 1]
        assert t.entry == d["Open"].iloc[i + 1]
        assert d["RSI"].iloc[i] < 35                      # le signal existe bien à la clôture du jour J


def test_gap_up_after_signal_is_paid():
    """Si le jour J+1 ouvre en gap haussier, l'entrée se fait au prix du gap, pas à la clôture de J."""
    df = make_ohlcv()
    d = prepare(df)
    k = int(np.flatnonzero(entry_signal(d, "RSI seul", 35).to_numpy())[0])
    df2 = df.copy()
    df2.iloc[k + 1, df2.columns.get_loc("Open")] = df2["Close"].iloc[k] * 1.10
    df2.iloc[k + 1, df2.columns.get_loc("High")] = df2["Open"].iloc[k + 1] * 1.01
    d2 = prepare(df2)
    first = simulate(d2, "RSI seul", P)[0].iloc[0]
    assert first["entry"] == pytest.approx(df["Close"].iloc[k] * 1.10)
    assert first["entry"] > d2["Close"].iloc[k]


def test_indicators_are_causal():
    df = make_ohlcv()
    full, cut = prepare(df), prepare(df.iloc[:500])
    for col in ["RSI", "ATR", "OBV_ok", "MACD_ok"]:
        pd.testing.assert_series_equal(full[col].iloc[:500], cut[col], check_names=False)
    for v in ["RSI seul", "RSI + OBV ou MACD", "RSI + OBV et MACD"]:
        pd.testing.assert_series_equal(entry_signal(full, v, 35).iloc[:500], entry_signal(cut, v, 35), check_names=False)


def test_future_prices_do_not_change_past_trades():
    df = make_ohlcv()
    t0 = 400
    poisoned = df.copy()
    rng = np.random.default_rng(99)
    poisoned.iloc[t0 + 1:] = poisoned.iloc[t0 + 1:] * rng.uniform(0.5, 1.5, (len(df) - t0 - 1, 1))
    a = simulate(prepare(df), "RSI seul", P)[0]
    b = simulate(prepare(poisoned), "RSI seul", P)[0]
    cutoff = df.index[t0 - 1]
    a, b = a[a["entry_date"] <= cutoff], b[b["entry_date"] <= cutoff]
    cols = ["signal", "entry_date", "entry"]
    assert len(a) > 0
    pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True))


def test_confirmations_match_app_labels():
    df = make_ohlcv(seed=3)
    d = prepare(df)
    for i in range(60, 400, 7):
        sub = d.iloc[:i + 1]
        obv_lbl = core.obv_signal(sub["Close"], sub["OBV"])
        macd_lbl = core.macd_state(float(sub["MACD_Hist"].iloc[-1]), float(sub["MACD_Hist"].iloc[-2]))
        assert bool(d["OBV_ok"].iloc[i]) == (obv_lbl in core.CONFIRMING_OBV)
        assert bool(d["MACD_ok"].iloc[i]) == (macd_lbl in core.CONFIRMING_MACD)


def test_variants_are_nested():
    d = prepare(make_ohlcv(seed=5))
    s0 = entry_signal(d, "RSI seul", 35)
    s1 = entry_signal(d, "RSI + OBV ou MACD", 35)
    s2 = entry_signal(d, "RSI + OBV et MACD", 35)
    assert (s1 <= s0).all() and (s2 <= s1).all() and s2.sum() < s0.sum()


def test_stop_fills_at_gap_open():
    df = make_ohlcv()
    d = prepare(df)
    k = int(np.flatnonzero(entry_signal(d, "RSI seul", 35).to_numpy())[0])
    e = k + 1
    stop = d["Open"].iloc[e] - 2.0 * d["ATR"].iloc[k]
    df2 = df.copy()
    gap = stop * 0.9
    df2.iloc[e + 1, df2.columns.get_loc("Open")] = gap
    df2.iloc[e + 1, df2.columns.get_loc("Low")] = gap * 0.99
    df2.iloc[e + 1, df2.columns.get_loc("High")] = gap * 1.01
    df2.iloc[e + 1, df2.columns.get_loc("Close")] = gap
    # on neutralise les autres sorties avant ce jour : seul le stop peut agir
    t = simulate(prepare(df2), "RSI seul", Params(35, None, 2.0, None, 0.0))[0].iloc[0]
    assert t["reason"] == "stop ATR" and t["exit_date"] == df2.index[e + 1]
    assert t["exit"] == pytest.approx(gap)               # exécuté à l'ouverture (gap), pas au niveau du stop


def test_fees_and_return_formula():
    d = prepare(make_ohlcv())
    t = simulate(d, "RSI seul", Params(35, 50, 2.0, 20, 0.001))[0].iloc[0]
    assert t["ret"] == pytest.approx(t["exit"] / t["entry"] * 0.999 / 1.001 - 1)


def test_equity_matches_compounded_trades():
    d = prepare(make_ohlcv(seed=7))
    trades, eq = simulate(d, "RSI seul", Params(35, 50, 2.0, 20, 0.001))
    assert eq.iloc[0] == pytest.approx(1.0)
    assert eq.iloc[-1] == pytest.approx(float((1 + trades["ret"]).prod()))


def test_split_and_aggregate_run():
    res = {"A": run_asset(make_ohlcv(seed=1), P), "B": run_asset(make_ohlcv(seed=2), P)}
    agg = aggregate(res)
    assert set(agg) == {"Construction (70 %)", "Hors échantillon (30 %)"}
    train_end = res["A"]["Construction (70 %)"]["RSI seul"]["equity"].index[-1]
    test_start = res["A"]["Hors échantillon (30 %)"]["RSI seul"]["equity"].index[0]
    assert train_end < test_start                          # aucune période commune
    assert agg["Hors échantillon (30 %)"]["RSI seul"]["stats"]["Trades"] >= 0


def test_verdict_flags_small_samples():
    few = pd.DataFrame({"ret": [0.05, -0.02, 0.03]})
    assert "Non concluant" in verdict(few, few)
    rng = np.random.default_rng(0)
    zero_mean = pd.DataFrame({"ret": rng.normal(0.0, 0.05, 60)})
    assert "hasard" in verdict(zero_mean, zero_mean + 0.0) or "Instable" in verdict(zero_mean, zero_mean)
    strong = pd.DataFrame({"ret": rng.normal(0.05, 0.01, 60)})
    assert "significatif" in verdict(strong, strong)
    flipped = pd.DataFrame({"ret": -strong["ret"]})
    assert "Instable" in verdict(strong, flipped)
