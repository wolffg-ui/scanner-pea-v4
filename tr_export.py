"""Calcul du PRU (prix de revient unitaire) à partir d'un export de transactions Trade Republic.

Usage :
    python tr_export.py "Exportation de transactions.csv"          # affiche le résultat
    python tr_export.py "Exportation de transactions.csv" --write  # écrit portfolio.json (ignoré par git)

Méthode : prix moyen pondéré. À l'achat, coût = montant + frais + taxe ; à la vente, la quantité vendue
sort au PRU courant (le PRU des parts restantes ne change pas). Les montants de l'export sont déjà en EUR.
"""
import argparse
import json
import sys

import pandas as pd

# Ligne de l'appli -> (compte, symbole/ISIN dans l'export)
EXPORT_SYMBOLS = {
    "MSCI Europe": ("PEA", "FR0013412038"),
    "Nasdaq 100": ("PEA", "FR0011871110"),
    "MSCI Emerging Markets": ("PEA", "FR0013412020"),
    "Veolia": ("PEA", "FR0000124141"),
    "ETH": ("DEFAULT", "ETH"),
    "XRP": ("DEFAULT", "XRP"),
    "TRX": ("DEFAULT", "TRX"),
}


def holdings_from_export(df: pd.DataFrame) -> dict:
    """{(compte, symbole): {"qty": quantité détenue, "cost": coût de revient restant (EUR)}}."""
    t = df[df["category"] == "TRADING"].copy()
    t = t[t["type"].isin(["BUY", "SELL"])].sort_values("datetime")
    for col in ("fee", "tax"):
        t[col] = pd.to_numeric(t[col], errors="coerce").fillna(0.0).abs()
    t["shares"] = pd.to_numeric(t["shares"], errors="coerce").abs()
    t["amount"] = pd.to_numeric(t["amount"], errors="coerce").abs()
    out = {}
    for (acc, sym), g in t.groupby(["account_type", "symbol"]):
        qty = cost = 0.0
        for r in g.itertuples():
            if r.type == "BUY":
                qty += r.shares
                cost += r.amount + r.fee + r.tax
            else:
                avg = cost / qty if qty > 0 else 0.0
                cost -= avg * r.shares
                qty -= r.shares
        if abs(qty) < 1e-9:
            qty = cost = 0.0
        out[(acc, sym)] = {"qty": qty, "cost": cost}
    return out


def costs_for_positions(df: pd.DataFrame, mapping: dict = EXPORT_SYMBOLS) -> dict:
    """{ligne: {"qty", "pru", "invested"}} pour les lignes de l'appli présentes dans l'export."""
    holdings = holdings_from_export(df)
    res = {}
    for line, key in mapping.items():
        h = holdings.get(key)
        if h and h["qty"] > 0:
            res[line] = {"qty": round(h["qty"], 8), "pru": round(h["cost"] / h["qty"], 6),
                         "invested": round(h["cost"], 2)}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--write", action="store_true", help="écrit portfolio.json")
    args = ap.parse_args(argv)
    costs = costs_for_positions(pd.read_csv(args.csv))
    for line, c in costs.items():
        print(f"{line:24} qté {c['qty']:>14.6f}  PRU {c['pru']:>12.4f} €  investi {c['invested']:>10.2f} €")
    if args.write:
        from core import save_costs
        save_costs({k: {"pru": v["pru"], "invested": v["invested"]} for k, v in costs.items()})
        print("portfolio.json écrit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
