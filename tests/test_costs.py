import pandas as pd
import pytest

import core
from tr_export import costs_for_positions, holdings_from_export


def tx(rows):
    cols = ["datetime", "account_type", "category", "type", "symbol", "shares", "amount", "fee", "tax"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])


def test_pru_includes_fee_and_tax():
    df = tx([("2026-01-01", "PEA", "TRADING", "BUY", "AAA", 10, -100.0, -1.0, -0.5)])
    h = holdings_from_export(df)[("PEA", "AAA")]
    assert h["qty"] == 10 and h["cost"] == pytest.approx(101.5)


def test_weighted_average_over_two_buys():
    df = tx([("2026-01-01", "PEA", "TRADING", "BUY", "AAA", 10, -100.0, 0, 0),
             ("2026-02-01", "PEA", "TRADING", "BUY", "AAA", 10, -200.0, 0, 0)])
    c = costs_for_positions(df, {"L": ("PEA", "AAA")})["L"]
    assert c["qty"] == 20 and c["pru"] == pytest.approx(15.0) and c["invested"] == pytest.approx(300.0)


def test_partial_sell_keeps_pru():
    df = tx([("2026-01-01", "DEFAULT", "TRADING", "BUY", "X", 10, -100.0, 0, 0),
             ("2026-02-01", "DEFAULT", "TRADING", "SELL", "X", -4, 80.0, -1.0, 0)])
    c = costs_for_positions(df, {"L": ("DEFAULT", "X")})["L"]
    assert c["qty"] == 6 and c["pru"] == pytest.approx(10.0)


def test_full_sell_removes_line():
    df = tx([("2026-01-01", "DEFAULT", "TRADING", "BUY", "X", 5, -50.0, 0, 0),
             ("2026-02-01", "DEFAULT", "TRADING", "SELL", "X", -5, 60.0, 0, 0)])
    assert costs_for_positions(df, {"L": ("DEFAULT", "X")}) == {}


def test_accounts_are_separate():
    df = tx([("2026-01-01", "DEFAULT", "TRADING", "BUY", "X", 2, -20.0, 0, 0),
             ("2026-01-02", "PEA", "TRADING", "BUY", "X", 3, -60.0, 0, 0)])
    c = costs_for_positions(df, {"L": ("PEA", "X")})["L"]
    assert c["qty"] == 3 and c["pru"] == pytest.approx(20.0)


def test_other_categories_ignored():
    df = tx([("2026-01-01", "PEA", "CASH", "DIVIDEND", "X", 2, 3.0, 0, 0),
             ("2026-01-02", "PEA", "TRADING", "BUY", "X", 1, -10.0, 0, 0)])
    assert holdings_from_export(df)[("PEA", "X")]["qty"] == 1


def test_apply_costs_and_real_pv(monkeypatch):
    pd_close = pd.Series([90.0, 95.0, 100.0], index=pd.date_range("2026-01-01", periods=3))
    monkeypatch.setattr(core, "fetch_close", lambda t, period="2y": pd_close)
    monkeypatch.setattr(core, "unit_price_eur", lambda line: 100.0)
    monkeypatch.setattr(core, "POSITIONS", {"L": {"ticker": "T", "qty": 10}})
    core.decision_table.clear()
    core.apply_costs({"L": {"pru": 80.0}})
    row = core.decision_table().iloc[0]
    assert row["PV latente"] == pytest.approx(0.25)
    assert row["PV (€)"] == pytest.approx(200.0)
    assert row["Investi (€)"] == pytest.approx(800.0) and bool(row["PV réelle"])
    core.apply_costs({})
    core.decision_table.clear()
    row = core.decision_table().iloc[0]
    assert not bool(row["PV réelle"]) and row["PV latente"] == pytest.approx(100 / 90 - 1)
    core.decision_table.clear()


def test_load_costs_from_env(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "PORTFOLIO_FILE", str(tmp_path / "absent.json"))
    monkeypatch.setenv("PORTFOLIO_JSON", '{"ETH": {"pru": 2000}}')
    assert core.load_costs() == {"ETH": {"pru": 2000}}
    core.save_costs({"X": {"pru": 1}})
    assert core.load_costs() == {"X": {"pru": 1}}   # le fichier prime
