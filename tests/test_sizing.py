import math

import pytest

from core import position_size


def size(**kw):
    base = dict(capital=10_000, risk_pct=1.0, price=100.0, stop=95.0, cap_pct=20.0, available=10_000.0, integer=True)
    base.update(kw)
    return position_size(**base)


def test_basic_formula():
    r = size()
    # risque 100 € / 5 € par titre = 20 titres = 2000 € = exactement le plafond de 20 %
    assert r["ok"] and r["qty"] == 20 and r["amount"] == pytest.approx(2000) and r["risk_eur"] == pytest.approx(100)
    assert r["limited_by"] is None


def test_line_cap_limits():
    r = size(stop=99.0)                      # risque seul → 100 titres (10 000 €) > plafond 2 000 €
    assert r["qty"] == 20 and r["limited_by"] == "plafond par ligne" and r["risk_eur"] == pytest.approx(20)


def test_envelope_limits():
    r = size(stop=99.0, available=500.0)
    assert r["qty"] == 5 and r["limited_by"] == "enveloppe disponible" and r["amount"] == pytest.approx(500)


def test_integer_floor_for_stocks_but_fractions_for_crypto():
    r_int = size(price=100.0, stop=97.0, capital=10_000, available=10_000, cap_pct=100)   # 100/3 = 33,33
    r_frac = size(price=100.0, stop=97.0, capital=10_000, available=10_000, cap_pct=100, integer=False)
    assert r_int["qty"] == 33
    assert r_frac["qty"] == pytest.approx(100 / 3)


def test_existing_holding_reduces_room():
    r = size(stop=99.0, current_value=1500.0)   # plafond 2 000 € − 1 500 € déjà détenus = 500 € → 5 titres
    assert r["qty"] == 5 and r["limited_by"] == "plafond par ligne"
    full = size(stop=99.0, current_value=2500.0)
    assert not full["ok"] and "marge" in full["reason"]


def test_less_than_one_share():
    r = size(price=3000.0, stop=2900.0, capital=10_000, cap_pct=20, available=10_000)   # 2 000 € < 1 titre à 3 000 €
    assert not r["ok"] and r["qty"] == 0 and "moins d'un titre" in r["reason"]


@pytest.mark.parametrize("kw,msg", [
    (dict(stop=float("nan")), "manquante"),
    (dict(price=float("nan")), "manquante"),
    (dict(stop=100.0), "stop au-dessus"),          # ATR nul → stop == prix
    (dict(stop=105.0), "stop au-dessus"),
    (dict(price=0.0), "prix invalide"),
    (dict(price=-1.0), "prix invalide"),
    (dict(capital=0.0), "capital nul"),
    (dict(capital=-5.0), "capital nul"),
    (dict(risk_pct=0.0), "risque"),
    (dict(available=0.0), "marge"),
])
def test_edge_cases_never_crash_and_return_zero(kw, msg):
    r = size(**kw)
    assert not r["ok"] and r["qty"] == 0 and r["amount"] == 0 and msg in r["reason"]


def test_risk_never_exceeds_target():
    for price, stop in [(100, 90), (7.3, 6.1), (0.12, 0.10), (2500, 2100)]:
        r = size(price=price, stop=stop, cap_pct=100, integer=False, available=1e9)
        assert r["risk_eur"] <= 100 + 1e-9 and math.isfinite(r["qty"])
