"""E-mail quotidien d'alertes (autonome, sans Streamlit).

Usage :
    python auto_alert.py --dry-run   # affiche l'e-mail sans l'envoyer
    python auto_alert.py             # envoie l'e-mail (RESEND_API_KEY requise)
"""
import argparse
import datetime
import logging
import sys

import resend

from core import (
    ALERT_RSI_HIGH, ALERT_RSI_LOW, CRYPTO_LINES, DEST_EMAIL, POSITIONS, SIZING_DEFAULTS, WATCHLIST_CRYPTO,
    WATCHLIST_STOCKS, YF_ANALYSIS_TICKERS, _get_secret, decision_table, fetch_fear_greed,
    next_earnings, portfolio_value, run_scan, signal_snapshot, size_signal,
)

log = logging.getLogger("auto_alert")
EARNINGS_WINDOW_DAYS = 14


def num(value: float) -> str:
    """Nombre avec 4 décimales sous 1 (petites cryptos), 2 sinon, espace comme séparateur de milliers."""
    return f"{value:,.{4 if abs(value) < 1 else 2}f}".replace(",", " ")


def money(value: float, ticker: str, kind: str) -> str:
    """Montant avec sa devise : USD pour les actions américaines, EUR sinon."""
    cur = "$" if kind == "Action" and not ticker.endswith(".PA") else "€"
    return f"{num(value)} {cur}"


def collect_signals() -> tuple[list, list]:
    """Retourne (dips RSI < seuil, surachats RSI > seuil) sur le portefeuille et la watchlist."""
    dips, overbought, seen = [], [], set()

    deci = decision_table()
    for _, r in deci.iterrows():
        entry = {"name": r["Ligne"], "ticker": r["Ticker"], "rsi": float(r["RSI(14)"]),
                 "kind": "Crypto" if r["Univers"] == "Crypto" else "PEA",
                 "pv": (float(r["PV latente"]), float(r["PV (€)"])) if r["PV réelle"] else None}
        if entry["rsi"] < ALERT_RSI_LOW:
            dips.append(entry)
        elif entry["rsi"] > ALERT_RSI_HIGH:
            overbought.append(entry)
        seen.add(r["Ticker"])

    scan, _ = run_scan()
    if not scan.empty:
        for _, r in scan[scan["dip"]].iterrows():
            yf_ticker = WATCHLIST_CRYPTO[r["Actif"]][1] if r["Type"] == "Crypto" else r["Actif"]
            if yf_ticker in seen:
                continue
            dips.append({"name": r["Actif"], "ticker": YF_ANALYSIS_TICKERS.get(yf_ticker, yf_ticker),
                         "rsi": float(r["rsi"]), "kind": r["Type"]})
    return dips, overbought


def enrich(entries: list) -> None:
    """Ajoute à chaque signal les indicateurs avancés ; en cas d'échec, le signal reste (sans détails)."""
    for e in entries:
        try:
            e["snap"] = signal_snapshot(e["ticker"])
        except Exception as ex:
            log.warning("Indicateurs indisponibles pour %s (%s) : %s", e["name"], e["ticker"], ex)
            e["snap"] = None


def format_line(e: dict, with_target: bool = True) -> str:
    s, t, k = e["snap"], e["ticker"], e["kind"]
    head = f"• {e['name']} ({t}) · RSI {e['rsi']:.0f}"
    if e.get("pv"):
        head += f" · PV réelle {e['pv'][0]:+.1%} ({e['pv'][1]:+,.0f} €)".replace(",", " ")
    if not s:
        return head + " · indicateurs avancés indisponibles"
    parts = [head, money(s["price"], t, k)]
    if with_target and s["target_rsi30"] > 0:
        parts.append(f"cible RSI 30 : {money(s['target_rsi30'], t, k)}")
    parts.append(f"Stop {money(s['stop'], t, k)} ({s['stop_pct']:+.1%})")
    parts.append(f"OBV {s['obv']}")
    parts.append(f"MACD {s['macd']}")
    sup, res = s["support"], s["resistance"]
    parts.append(f"Support {num(sup[1])} ({sup[0]})" if sup else "Support —")
    parts.append(f"Résistance {num(res[1])} ({res[0]})" if res else "Résistance —")
    sz = e.get("size")
    if sz is not None:
        if sz["ok"]:
            qty = f"{sz['qty']:.0f}" if k != "Crypto" else f"{sz['qty']:.4f}".rstrip("0").rstrip(".")
            lim = f" ; limité : {sz['limited_by']}" if sz["limited_by"] else ""
            parts.append(f"Taille : {qty} ≈ {num(sz['amount'])} € (risque {num(sz['risk_eur'])} €{lim})")
        else:
            parts.append(f"Taille : n/a ({sz['reason']})")
    return " · ".join(parts)


def earnings_warnings() -> list:
    """Actifs du portefeuille et de la watchlist dont les résultats tombent dans les 14 jours."""
    tickers = {pos["ticker"]: line for line, pos in POSITIONS.items() if line not in CRYPTO_LINES}
    tickers.update({t: t for ts in WATCHLIST_STOCKS.values() for t in ts})
    today, out = datetime.date.today(), []
    for ticker, name in tickers.items():
        try:
            d = next_earnings(ticker)
        except Exception as ex:
            log.warning("Résultats indisponibles pour %s : %s", ticker, ex)
            continue
        if d and 0 <= (d - today).days <= EARNINGS_WINDOW_DAYS:
            out.append((d, f"📅 Résultats dans {(d - today).days} jours — {name} ({ticker}) le {d:%d/%m/%Y}"))
    return [txt for _, txt in sorted(out)]


def build_report() -> tuple[str, str, int]:
    """Retourne (sujet, corps, nombre de signaux)."""
    dips, overbought = collect_signals()
    enrich(dips)
    enrich(overbought)
    try:
        capital = portfolio_value()
    except Exception as ex:
        log.warning("Valeur du portefeuille indisponible (%s) : taille de position omise", ex)
        capital = 0.0
    for e in dips:
        if e["snap"]:
            e["size"] = size_signal(e["snap"], e["ticker"], e["kind"], capital, SIZING_DEFAULTS["risk_pct"],
                                    SIZING_DEFAULTS["cap_pct"], SIZING_DEFAULTS["budget"])
    confirmed = [e for e in dips if e["snap"] and e["snap"]["dip_confirmed"]]
    watch = [e for e in dips if e not in confirmed]

    lines = [f"📊 Rapport d'opportunités — {datetime.date.today():%d/%m/%Y}", ""]
    fg = fetch_fear_greed()
    if fg:
        week = f" ({fg['value'] - fg['week_ago']:+d} sur 7 jours)" if fg["week_ago"] is not None else ""
        lines += [f"😨 Fear & Greed crypto : {fg['value']}/100 · {fg['label']}{week}", ""]

    def section(title, entries, **kw):
        if entries:
            lines.extend([title, *(format_line(e, **kw) for e in entries), ""])

    section("🔥 Dip confirmé (OBV ou MACD favorable)", confirmed)
    section("👀 À surveiller (non confirmé)", watch)
    section("🔴 Surachat", overbought, with_target=False)

    warnings = earnings_warnings()
    if warnings:
        lines += ["⚠️ Résultats à venir", *warnings, ""]

    lines.append("Confirmation : OBV 🟢/🟡 ou MACD 🟢/🟡. Stop = prix − 2×ATR(14). Supports/résistances : points pivots.")
    lines.append(f"Taille : risque {SIZING_DEFAULTS['risk_pct']:g} % du portefeuille ({num(capital)} €), "
                 f"plafond {SIZING_DEFAULTS['cap_pct']:g} % par ligne, enveloppe {num(SIZING_DEFAULTS['budget'])} € par signal.")
    n = len(dips) + len(overbought)
    return "⚡ Rapport Automatique d'Opportunités Bourse", "\n".join(lines), n


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="affiche l'e-mail sans l'envoyer")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # les 404 « calendar » des ETF/cryptos sont normaux

    subject, body, n = build_report()
    if n == 0:
        print("Aucun signal critique aujourd'hui.")
        return 0
    if args.dry_run:
        print(f"[DRY-RUN] À : {DEST_EMAIL}\nObjet : {subject}\n\n{body}")
        return 0

    api_key = _get_secret("RESEND_API_KEY")
    if not api_key:
        log.error("RESEND_API_KEY introuvable (variable d'environnement ou .streamlit/secrets.toml).")
        return 1
    resend.api_key = api_key
    try:
        resend.Emails.send({"from": "BOURSE APP <onboarding@resend.dev>", "to": DEST_EMAIL,
                            "subject": subject, "text": body})
    except Exception as ex:
        log.error("Échec de l'envoi Resend : %s", ex)
        return 1
    print(f"Alerte envoyée à {DEST_EMAIL} ({n} signal(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
