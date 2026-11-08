"""Scanner d'opportunités & assistant d'allocation (PEA + CTO/Crypto).

Lancer : streamlit run app.py
"""
import hmac
import os
import requests
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
import resend

# Configuration de la page (DOIT être la première commande Streamlit)
st.set_page_config(page_title="Scanner d'opportunités", page_icon="📈", layout="wide")

# ----------------------------------------------------------------------------
# Écran de connexion
# ----------------------------------------------------------------------------
APP_PASSWORD = "CryptoPEA2026"

if not st.session_state.get("authenticated"):
    pwd = st.text_input("Mot de passe", type="password")
    if pwd and hmac.compare_digest(pwd, APP_PASSWORD):
        st.session_state["authenticated"] = True
        st.rerun()
    st.warning("Accès restreint")
    if pwd:
        st.error("Mot de passe incorrect")
    st.stop()

from core import *  # noqa: F401,F403,E402
from core import _get_secret  # noqa: F401,E402


# ----------------------------------------------------------------------------
# Allocation & DCA
# ----------------------------------------------------------------------------
def allocate_core(envelope: float, current: dict, targets: dict) -> pd.DataFrame:
    tw = pd.Series(targets, dtype=float)
    tw = tw / tw.sum()
    cur = pd.Series(current, dtype=float)[tw.index]
    total_after = cur.sum() + envelope
    deficit = (tw * total_after - cur).clip(lower=0)
    if deficit.sum() > envelope: buy = deficit / deficit.sum() * envelope
    else: buy = deficit + tw * (envelope - deficit.sum())
    after = cur + buy
    return pd.DataFrame({
        "Ligne": tw.index, "Valeur actuelle (€)": cur.values,
        "Poids actuel": (cur / cur.sum()).fillna(0).values if cur.sum() else 0.0,
        "Poids cible": tw.values, "À investir (€)": buy.values,
        "Poids après": (after / total_after).values,
    })

def allocate_opportunities(envelope: float, signals: pd.DataFrame, max_n: int = 3) -> pd.DataFrame:
    if signals.empty or envelope <= 0: return pd.DataFrame()
    s = signals.copy()
    s["score"] = np.where(s["dip"], 1.0 + (RSI_THRESHOLD - s["rsi"]) / RSI_THRESHOLD, 0.5 + s["vol_ratio"] / 10)
    s = s.sort_values("score", ascending=False).head(max_n)
    s["Montant (€)"] = s["score"] / s["score"].sum() * envelope
    s["Signal"] = np.where(s["dip"], "🟢 Dip", "🔵 Breakout")
    return s

DCA_MESSAGES = {
    ("ETF", "block"): "🔴 Surachat ETF (RSI > 70) — Achat temporairement bloqué",
    ("Crypto", "block"): "🔴 Surchauffe Crypto (RSI > 75) — Achat temporairement bloqué",
    ("ETF", "boost"): "🟢 Opportunité ETF (RSI < 40) — Surpondérer l'achat",
    ("Crypto", "boost"): "🟢 Opportunité Crypto (RSI < 35) — Surpondérer l'achat",
}
DCA_BOOST = 0.5

def smart_dca_allocate(envelope: float, current: dict, targets: dict, info: dict) -> pd.DataFrame:
    base = allocate_core(envelope, current, targets)
    idx = list(base["Ligne"])
    buy = pd.Series(base["À investir (€)"].to_numpy(), index=idx, dtype=float)
    rsi = pd.Series({l: info.get(l, {}).get("rsi", float("nan")) for l in idx})
    trend = pd.Series({l: info.get(l, {}).get("trend", "—") for l in idx})
    is_crypto = pd.Series({l: l in CRYPTO_LINES for l in idx})

    blocked = (~is_crypto & (rsi > 70)) | (is_crypto & (rsi > 75))
    boosted = (~is_crypto & (rsi < 40)) | (is_crypto & (rsi < 35))

    freed = float(buy.where(blocked, 0.0).sum())
    buy[blocked] = 0.0
    if freed > 0:
        cand = ~blocked
        rsi_pool = rsi.where(cand) if cand.any() else rsi
        if rsi_pool.notna().any():
            recip = (rsi == rsi_pool.min())
            if cand.any(): recip = recip & cand
        else: recip = cand if cand.any() else pd.Series(True, index=idx)
        n = int(recip.sum())
        if n > 0: buy[recip] = buy[recip] + freed / n

    if boosted.any():
        extra = float((buy.where(boosted, 0.0) * DCA_BOOST).sum())
        donors = ~boosted & ~blocked
        donor_amt = buy.where(donors, 0.0)
        if extra > 0 and donor_amt.sum() >= extra:
            buy[boosted] = buy[boosted] * (1 + DCA_BOOST)
            buy[donors] = buy[donors] - extra * donor_amt[donors] / donor_amt.sum()

    def message(line: str) -> str:
        cls = "Crypto" if line in CRYPTO_LINES else "ETF"
        if blocked[line]: return DCA_MESSAGES[(cls, "block")]
        if boosted[line]: return DCA_MESSAGES[(cls, "boost")]
        return ""

    cur = pd.Series(base["Valeur actuelle (€)"].to_numpy(), index=idx, dtype=float)
    total_after = cur.sum() + envelope
    return pd.DataFrame({
        "Ligne": idx, "Valeur actuelle (€)": cur.to_numpy(),
        "Poids actuel": base["Poids actuel"].to_numpy(), "Poids cible": base["Poids cible"].to_numpy(),
        "RSI(14)": rsi.to_numpy(), "Tendance": trend.to_numpy(),
        "À investir (€)": buy.to_numpy(), "Poids après": ((cur + buy) / total_after).to_numpy() if total_after else 0.0,
        "Conseil DCA": [message(l) for l in idx],
    })


# ----------------------------------------------------------------------------
# Analyse approfondie (UI)
# ----------------------------------------------------------------------------
def analysis_universe() -> dict:
    """Actifs analysables : lignes du portefeuille + watchlist -> (ticker yfinance, type)."""
    opts = {}
    for line, pos in POSITIONS.items():
        opts[f"{line} · portefeuille"] = (pos["ticker"], "Crypto" if line in CRYPTO_LINES else "PEA")
    for grp, tickers in WATCHLIST_STOCKS.items():
        for t in tickers:
            opts[f"{t} · {grp}"] = (t, "Action")
    for sym, (_, yf_ticker) in WATCHLIST_CRYPTO.items():
        opts[f"{sym} · watchlist crypto"] = (YF_ANALYSIS_TICKERS.get(yf_ticker, yf_ticker), "Crypto")
    return opts


def render_deep_analysis(ticker: str, kind: str, atr_mult: float, sr_method: str):
    try:
        d = add_advanced_columns(fetch_ohlcv(ticker))
        if len(d) < 60:
            raise ValueError("historique insuffisant")
    except Exception as e:
        st.error(f"Données indisponibles pour {ticker} : {e}")
        return

    if d["Volume"].tail(10).median() == 0:
        st.warning(f"⚠️ Les données Yahoo de {ticker} semblent figées (volume nul sur les derniers jours) : "
                   "ATR, Stop-Loss et niveaux ne sont pas fiables.")

    last, prev = d.iloc[-1], d.iloc[-2]
    price, atr = float(last["Close"]), float(last["ATR"])
    stop = price - atr_mult * atr
    levels = pivot_levels(d) if sr_method == "Points pivots" else fib_levels(d)
    support, resistance = nearest_levels(levels, price)

    bb_range = float(last["BB_up"] - last["BB_low"])
    pct_b = float((price - last["BB_low"]) / bb_range) if bb_range > 0 else float("nan")
    bandwidth = ((d["BB_up"] - d["BB_low"]) / d["BB_mid"]).dropna()
    squeeze = len(bandwidth) > 30 and bool(bandwidth.iloc[-1] <= bandwidth.tail(120).quantile(0.10))
    obv_lbl = obv_signal(d["Close"], d["OBV"])
    hist_now, hist_prev = float(last["MACD_Hist"]), float(prev["MACD_Hist"])
    macd_lbl = macd_state(hist_now, hist_prev)
    mm200 = float(last["MM200"])

    def card(col, title, value, text, help=None):
        """Carte lisible : titre, verdict (retour à la ligne autorisé), phrase d'explication."""
        with col.container(border=True):
            st.caption(f"**{title}**", help=help)
            st.markdown(f"**{value}**")
            st.caption(text)

    cur = " $" if kind == "Action" and not ticker.endswith(".PA") else " €"

    # --- Ligne 1 : l'avis des 4 indicateurs (un verdict + une phrase chacun) ---
    st.markdown("#### Que disent les indicateurs ?")
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    card(c1, "Volume (OBV)", obv_lbl, OBV_HELP[obv_lbl],
         help="L'OBV cumule les volumes échangés les jours de hausse et de baisse : il vérifie que les volumes accompagnent le prix.")
    card(c2, "Volatilité (Bollinger)", bollinger_label(pct_b),
         f"Le prix est à {pct_b:.0%} de la fourchette (0 % = bande basse, 100 % = bande haute)." if pct_b == pct_b else "Pas assez de données.")
    card(c3, "Momentum (MACD)", macd_lbl, MACD_HELP[macd_lbl],
         help="Pente de l'histogramme MACD : elle change avant les moyennes mobiles, donc anticipe les retournements.")
    card(c4, "Tendance (MM200)", trend_signal(price, mm200),
         f"Le prix est {abs(price / mm200 - 1):.1%} {'au-dessus' if price > mm200 else 'en dessous'} de sa moyenne sur 200 jours." if mm200 == mm200 else "Moins de 200 jours d'historique.")
    if squeeze:
        st.info("⚡ Compression des Bandes de Bollinger : la volatilité est à un point bas, un mouvement fort peut se préparer (sans indiquer la direction).")

    # --- Ligne 2 : risque et niveaux clés ---
    st.markdown("#### Risque et niveaux clés")
    r1, r2 = st.columns(2)
    r3, r4 = st.columns(2)
    card(r1, f"🛑 Stop-Loss ({atr_mult:g}×ATR)", f"{stop:,.2f}{cur}",
         f"Soit {abs(stop / price - 1):.1%} sous le prix actuel. Vendre si le cours descend sous ce niveau limite la perte.",
         help=f"Prix − {atr_mult:g} × ATR({ATR_PERIOD}). Un stop trop serré sera touché par le bruit normal du marché.")
    card(r2, f"Volatilité du jour (ATR {ATR_PERIOD}j)", f"{atr:,.2f}{cur}",
         f"Le cours bouge en moyenne de {atr / price:.1%} par jour.",
         help="ATR = Average True Range : amplitude moyenne journalière.")
    card(r3, "Support le plus proche", f"{support[1]:,.2f}{cur}" if support else "—",
         f"{support[0]} · {abs(support[1] / price - 1):.1%} sous le prix. Zone où la baisse a déjà été freinée." if support else "Aucun niveau sous le prix.")
    card(r4, "Résistance la plus proche", f"{resistance[1]:,.2f}{cur}" if resistance else "—",
         f"{resistance[0]} · {abs(resistance[1] / price - 1):.1%} au-dessus du prix. Zone où la hausse a déjà été freinée." if resistance else "Aucun niveau au-dessus du prix.")

    # --- Contexte macro (selon le type d'actif) ---
    st.markdown("#### Contexte")
    m1, _ = st.columns(2)
    if kind == "Crypto":
        fg = fetch_fear_greed()
        if fg:
            wk = f" Il y a 7 jours : {fg['week_ago']}/100." if fg["week_ago"] is not None else ""
            card(m1, "😨 Fear & Greed (marché crypto)", f"{fg['value']}/100 · {fg['label']}",
                 "0 = peur extrême, 100 = avidité extrême." + wk)
            if fg["value"] <= 25:
                st.caption("Peur extrême : zone souvent associée à des phases d'accumulation, mais aussi à une forte volatilité.")
            elif fg["value"] >= 75:
                st.caption("Avidité extrême : zone de prudence, les excès de marché se corrigent souvent brutalement.")
        else:
            st.caption("Fear & Greed Index indisponible pour le moment.")
    else:
        earn = next_earnings(ticker)
        if earn:
            days = (earn - datetime.date.today()).days
            card(m1, "📅 Prochains résultats", earn.strftime("%d/%m/%Y"), f"Dans {days} jours.")
            if days <= 14:
                st.warning("Résultats imminents : forte volatilité possible, à garder en tête avant d'acheter ou de placer un stop.")
        else:
            st.caption("📅 Pas de date de résultats disponible (ETF ou donnée indisponible).")

    # --- Graphique unique : prix + Bollinger + niveaux / OBV / MACD ---
    view = d.tail(120)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.55, 0.2, 0.25])
    fig.add_trace(go.Scatter(x=view.index, y=view["BB_up"], line=dict(width=0), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["BB_low"], line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(37,99,235,0.10)", hoverinfo="skip", name="Bollinger"), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["Close"], mode="lines", name="Prix", line=dict(color="#2563eb", width=2)), row=1, col=1)
    if view["MM200"].notna().any():
        fig.add_trace(go.Scatter(x=view.index, y=view["MM200"], mode="lines", name="MM200",
                                 line=dict(color="#9ca3af", width=1, dash="dash")), row=1, col=1)
    fig.add_hline(y=stop, line_dash="dash", line_color="#dc2626", annotation_text=f"Stop {stop:,.2f}", row=1, col=1)
    if support:
        fig.add_hline(y=support[1], line_dash="dot", line_color="#16a34a", annotation_text=support[0], row=1, col=1)
    if resistance:
        fig.add_hline(y=resistance[1], line_dash="dot", line_color="#f59e0b", annotation_text=resistance[0], row=1, col=1)

    fig.add_trace(go.Scatter(x=view.index, y=view["OBV"], mode="lines", name="OBV", line=dict(color="#7c3aed")), row=2, col=1)

    hist_colors = ["#16a34a" if v >= 0 else "#dc2626" for v in view["MACD_Hist"]]
    fig.add_trace(go.Bar(x=view.index, y=view["MACD_Hist"], marker_color=hist_colors, name="Histogramme"), row=3, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["MACD"], mode="lines", name="MACD", line=dict(color="#2563eb", width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["Signal_Line"], mode="lines", name="Signal", line=dict(color="#f97316", width=1)), row=3, col=1)

    fig.update_yaxes(title_text="Prix", row=1, col=1)
    fig.update_yaxes(title_text="OBV", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_layout(height=640, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander(f"Voir tous les niveaux ({sr_method})"):
        lv = pd.DataFrame({"Niveau": list(levels.keys()), "Prix": list(levels.values())})
        lv["Écart vs prix"] = lv["Prix"] / price - 1
        st.dataframe(lv.sort_values("Prix", ascending=False).style.format({"Prix": "{:,.2f}", "Écart vs prix": "{:+.1%}"}),
                     hide_index=True, use_container_width=True)
    st.caption("Prix dans la devise de cotation : EUR pour les ETF/actions françaises et les cryptos (converties), USD pour les actions américaines.")


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("💶 Budget du mois")
    budget = st.slider("Budget à investir (€)", 100, 500, 500, step=25)
    core_pct = st.slider("Cœur PEA (%)", 0, 100, 60, step=5)
    st.caption(f"Cœur PEA **{core_pct}%** · Opportunités **{100 - core_pct}%**")

    with st.form("holdings"):
        st.subheader("Valeur actuelle des lignes")
        st.markdown("**PEA**")
        pea_values = {n: st.number_input(f"{n} (€)", min_value=0.0, value=position_value(n), step=50.0) for n in PEA_LINES}
        st.markdown("**Crypto**")
        crypto_values = {n: st.number_input(f"{n} (€)", min_value=0.0, value=position_value(n), step=50.0) for n in CRYPTO_LINES}
        st.markdown("**Allocation cible PEA (%)**")
        targets = {n: st.number_input(f"Cible {n}", 0, 100, DEFAULT_TARGETS[n], step=5) for n in PEA_LINES}
        st.form_submit_button("Mettre à jour", use_container_width=True)

    st.divider()
    st.subheader("📩 Service d'alertes")
    st.caption(f"Expéditeur : **BOURSE APP**\nDestinataire : **{DEST_EMAIL}**")

    if st.button("📩 Tester l'envoi d'un e-mail d'alerte", use_container_width=True):
        ok, detail = send_resend_email(
            "Test d'alerte — Scanner d'opportunités",
            "✅ Ceci est un e-mail de test configuré automatiquement avec ton compte Resend."
        )
        (st.success if ok else st.error)(detail)

core_env = budget * core_pct / 100
opp_env = budget - core_env

# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.title("📈 Scanner d'opportunités & allocation")
c1, c2, c3 = st.columns(3)
c1.metric("Budget du mois", f"{budget:,.0f} €")
c2.metric("Enveloppe Cœur PEA", f"{core_env:,.0f} €", f"{core_pct}%")
c3.metric("Enveloppe Opportunités", f"{opp_env:,.0f} €", f"{100 - core_pct}%")

tab_core, tab_scan, tab_opp, tab_decision, tab_deep, tab_bilan = st.tabs(
    ["🏛️ Cœur PEA", "🔍 Scanner", "🎯 Opportunités", "🧭 Aide à la décision", "🔬 Analyse approfondie", "🧾 Bilan & Fiscalité"])

with tab_core:
    if sum(targets.values()) == 0:
        st.error("Les cibles PEA sont toutes à 0.")
    else:
        if sum(targets.values()) != 100:
            st.warning(f"Les cibles totalisent {sum(targets.values())}% : elles sont normalisées à 100%.")
        core_df = allocate_core(core_env, pea_values, targets)
        cols = st.columns(len(core_df))
        for col, (_, r) in zip(cols, core_df.iterrows()):
            col.metric(r["Ligne"], f"{r['À investir (€)']:,.0f} €",
                       f"{(r['Poids après'] - r['Poids actuel']) * 100:+.1f} pts de poids")
        st.dataframe(
            core_df.style.format({
                "Valeur actuelle (€)": "{:,.0f}", "À investir (€)": "{:,.0f}",
                "Poids actuel": "{:.1%}", "Poids cible": "{:.1%}", "Poids après": "{:.1%}",
            }).background_gradient(subset=["À investir (€)"], cmap="Greens"),
            hide_index=True, use_container_width=True,
        )

with tab_decision:
    st.subheader("📊 Signaux techniques — RSI 14j & Tendance MM50/MM200")
    with st.spinner("Calcul des indicateurs…"): deci = decision_table()
    if deci.empty: st.error("Aucune donnée récupérée pour les lignes détenues (réseau / yfinance).")
    else:
        st.dataframe(
            deci[["Univers", "Ligne", "Ticker", "Prix", "RSI(14)", "Conseil RSI", "MM50", "MM200", "Tendance"]].style.format({
                "Prix": "{:,.2f}", "RSI(14)": "{:.1f}", "MM50": "{:,.2f}", "MM200": "{:,.2f}"}),
            hide_index=True, use_container_width=True,
        )
    st.divider()
    st.subheader("🛡️ Prise de profit & Sécurisation")
    if not deci.empty:
        overheated = pd.Series(np.where(deci["Univers"] == "Crypto", deci["RSI(14)"] > 75, deci["RSI(14)"] > 70), index=deci.index)
        secure = deci[(deci["PV latente"] > 0.30) & overheated]
        if secure.empty: st.success("Aucune ligne à sécuriser.")
        else:
            st.dataframe(secure[["Univers", "Ligne", "Ticker", "Prix", "PV latente", "RSI(14)"]].style.format(
                    {"Prix": "{:,.2f}", "PV latente": "{:+.1%}", "RSI(14)": "{:.1f}"}), hide_index=True, use_container_width=True)
    st.divider()
    st.subheader("💸 Smart DCA — rebalancement & conseils conditionnels")
    lines = list(POSITIONS.keys())
    dca_amount = st.number_input("Montant mensuel (€)", min_value=0.0, value=200.0, step=50.0)
    dca_targets, cols = {}, st.columns(len(lines))
    for col, line in zip(cols, lines):
        dca_targets[line] = col.number_input(line, min_value=0.0, value=float(DCA_DEFAULT_TARGETS.get(line, round(100 / len(lines), 1))), step=1.0)

    if dca_amount > 0 and sum(dca_targets.values()) > 0:
        current = {line: position_value(line) for line in lines}
        info = ({r["Ligne"]: {"rsi": r["RSI(14)"], "trend": r["Tendance"]} for _, r in deci.iterrows()} if not deci.empty else {})
        dca = smart_dca_allocate(dca_amount, current, dca_targets, info)
        dca["Prix unitaire (€)"] = dca["Ligne"].map(unit_price_eur)
        dca["Qté à acheter"] = (dca["À investir (€)"] / dca["Prix unitaire (€)"].replace(0, np.nan)).fillna(0)
        st.dataframe(
            dca[["Ligne", "RSI(14)", "Tendance", "À investir (€)", "Prix unitaire (€)", "Qté à acheter", "Conseil DCA"]].style.format({
                "RSI(14)": "{:.1f}", "À investir (€)": "{:,.2f}", "Prix unitaire (€)": "{:,.2f}", "Qté à acheter": "{:,.4f}"
            }).background_gradient(subset=["Qté à acheter"], cmap="Greens"),
            hide_index=True, use_container_width=True,
        )

with tab_deep:
    st.subheader("🔬 Analyse approfondie d'un actif")
    universe = analysis_universe()
    sel_col, set_col = st.columns([2, 1])
    choice = sel_col.selectbox("Actif à analyser", list(universe.keys()))
    with set_col.expander("⚙️ Réglages"):
        atr_mult = st.number_input("Multiple d'ATR (Stop-Loss)", min_value=0.5, max_value=5.0, value=ATR_MULT, step=0.5)
        sr_method = st.radio("Supports / Résistances", ["Points pivots", "Fibonacci"], horizontal=True)
    deep_ticker, deep_kind = universe[choice]
    render_deep_analysis(deep_ticker, deep_kind, atr_mult, sr_method)

with tab_bilan:
    st.subheader("🧾 Bilan & Fiscalité")
    
    # --- Module Historique du Portefeuille (Évolution 4) ---
    st.markdown("### 📈 Suivi Historique du Portefeuille")
    total_pea = sum(pea_values.values())
    total_crypto = sum(crypto_values.values())
    
    col_hist1, col_hist2 = st.columns([1, 3])
    with col_hist1:
        st.metric("Total PEA", f"{total_pea:,.0f} €")
        st.metric("Total Crypto", f"{total_crypto:,.0f} €")
        st.metric("Valeur Totale", f"{total_pea + total_crypto:,.0f} €")
        if st.button("📸 Sauvegarder ce point historique", use_container_width=True):
            save_portfolio_snapshot(total_pea, total_crypto)
            st.success("Point historique enregistré !")
            
    with col_hist2:
        hist_df = load_portfolio_history()
        if not hist_df.empty:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["PEA"], mode="lines+markers", name="PEA", line=dict(color="#2563eb")))
            fig_hist.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["Crypto"], mode="lines+markers", name="Crypto", line=dict(color="#f59e0b")))
            fig_hist.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["Total"], mode="lines+markers", name="Total", line=dict(color="#16a34a", width=3)))
            fig_hist.update_layout(title="Progression du Portefeuille (€)", height=250, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Clique sur le bouton pour enregistrer la première mesure de ton portefeuille.")

    st.divider()
    st.markdown("### 📊 Plafond PEA")
    versements = st.number_input("Versements cumulés sur le PEA (€)", min_value=0.0, max_value=1_000_000.0, value=float(round(total_pea)), step=500.0)
    PLAFOND_PEA = 150_000.0
    b1, b2, b3 = st.columns(3)
    b1.metric("Versements", f"{versements:,.0f} €")
    b2.metric("Plafond", f"{PLAFOND_PEA:,.0f} €")
    b3.metric("Disponible", f"{max(PLAFOND_PEA - versements, 0.0):,.0f} €")
    st.progress(min(versements / PLAFOND_PEA, 1.0))
    st.divider()
    st.markdown("### 📩 Alertes RSI")
    alerts = rsi_alerts(deci)
    if not alerts:
        st.success("Aucun actif en opportunité ou en surachat actuellement.")
    else:
        for a in alerts: st.write("- " + a)
        if st.button("📩 Envoyer ces alertes par e-mail", use_container_width=True):
            ok, detail = send_resend_email("Alertes RSI — Scanner d'opportunités", "📊 Alertes RSI :\n" + "\n".join(alerts))
            (st.success if ok else st.error)(detail)

with st.spinner("Scan des marchés en cours…"):
    scan, histories = run_scan()

if scan.empty:
    st.stop()

signals = scan[scan["dip"] | scan["breakout"]]

with tab_scan:
    view = scan.assign(Signal=np.select([scan.dip, scan.breakout], ["🟢 Dip", "🔵 Breakout"], "—"))
    st.caption("Prix Cible RSI 30 : Cours théorique à atteindre pour placer un ordre d'achat limite en zone d'opportunité. "
               "Volume (OBV) : le volume confirme-t-il la tendance ? Momentum (MACD) : pente de l'histogramme. "
               "Détails, Stop-Loss ATR, Bollinger et niveaux clés dans l'onglet 🔬 Analyse approfondie.")
    st.dataframe(
        view[["Actif", "Univers", "price", "rsi", "target_rsi30", "macd_state", "obv_signal", "high20", "vol_ratio", "perf_60d", "Signal"]].rename(columns={
            "price": "Prix", "rsi": "RSI(14)", "target_rsi30": "Prix Cible (RSI 30)", "macd_state": "Momentum (MACD)", "obv_signal": "Volume (OBV)", f"high20": f"Plus haut {LOOKBACK}j",
            "vol_ratio": "Volume / moy.", "perf_60d": "Perf 60j"}).style.format({
            "Prix": "{:,.2f}", "RSI(14)": "{:.1f}", "Prix Cible (RSI 30)": "{:,.2f}", f"Plus haut {LOOKBACK}j": "{:,.2f}",
            "Volume / moy.": "{:.2f}×", "Perf 60j": "{:+.1%}"}),
        hide_index=True, use_container_width=True,
    )

with tab_opp:
    if not signals.empty:
        alloc = allocate_opportunities(opp_env, signals)
        cols = st.columns(len(alloc))
        for col, (_, r) in zip(cols, alloc.iterrows()):
            col.metric(f"{r['Signal']} · {r['Actif']}", f"{r['Montant (€)']:,.0f} €", f"RSI {r['rsi']:.0f} · {r['perf_60d']:+.1%} /60j", delta_color="normal" if r["dip"] else "off")
        
        # --- Visualisation Graphique MACD & Hist (Évolution 1) ---
        for _, r in signals.iterrows():
            df = histories[r["Actif"]]
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # Subplot 1: Prix
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Prix", line=dict(color="#16a34a" if r["dip"] else "#2563eb")), row=1, col=1)
            fig.add_hline(y=r["high20"], line_dash="dot", annotation_text=f"Plus haut {LOOKBACK}j", row=1, col=1)
            
            # Subplot 2: MACD + Signal + Histogramme
            fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD", line=dict(color="#2563eb")), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["Signal_Line"], mode="lines", name="Signal", line=dict(color="#f97316")), row=2, col=1)
            
            colors = ["#16a34a" if val >= 0 else "#dc2626" for val in df["MACD_Hist"]]
            fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogramme", marker_color=colors), row=2, col=1)
            
            fig.update_layout(title=f"{r['Actif']} — Courbe Prix & Indicateur MACD", height=400, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
