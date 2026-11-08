"""Données, indicateurs et alertes (sans interface Streamlit), partagés par app.py et auto_alert.py."""
import json
import math
import os
import requests
import datetime
import functools
import logging
import time
import numpy as np
import pandas as pd
import yfinance as yf
import resend

log = logging.getLogger("core")


def cached(ttl: float):
    """Cache mémoire simple avec durée de vie (remplace st.cache_data, sans Streamlit).

    Renvoie des copies des DataFrames/Series pour que l'appelant puisse les modifier.
    Les exceptions ne sont pas mises en cache.
    """
    def deco(fn):
        store = {}

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            hit = store.get(key)
            if hit is None or time.time() - hit[0] > ttl:
                hit = (time.time(), fn(*args, **kwargs))
                store[key] = hit
            val = hit[1]
            return val.copy() if hasattr(val, "copy") else val
        wrapper.clear = store.clear
        return wrapper
    return deco


# ----------------------------------------------------------------------------
# Univers
# ----------------------------------------------------------------------------
WATCHLIST_STOCKS = {
    "Tech": ["NVDA", "MSFT", "AAPL"],
    "Value/Dividendes": ["JNJ", "KO", "PG", "O", "PEP"],
    "Eau": ["AWAT.PA", "XYL", "AWK"],
    "Terres Rares": ["REMX", "ALB"]
}

WATCHLIST_CRYPTO = {
    "SOL": ("solana", "SOL-USD"),
    "AVAX": ("avalanche-2", "AVAX-USD"),
    "LINK": ("chainlink", "LINK-USD"),
    "ONDO": ("ondo-finance", "ONDO-USD"),
    "FET": ("fetch-ai", "FET-USD"),
    "ARB": ("arbitrum", "ARB-USD"),
}
PEA_LINES = ["MSCI Europe", "Nasdaq 100", "MSCI Emerging Markets", "Veolia"]
CRYPTO_LINES = ["ETH", "XRP", "TRX"]
DEFAULT_TARGETS = {"MSCI Europe": 30, "Nasdaq 100": 30, "MSCI Emerging Markets": 20, "Veolia": 20}

POSITIONS = {
    "MSCI Europe": {"ticker": "PCEU.PA", "qty": 18, "target_eur": 584.55},
    "Nasdaq 100": {"ticker": "PUST.PA", "qty": 7, "target_eur": 654.72},
    "MSCI Emerging Markets": {"ticker": "PAEEM.PA", "qty": 36},
    "Veolia": {"ticker": "VIE.PA", "qty": 7, "target_eur": 91.83},
    "ETH": {"ticker": "ETH-USD", "qty": 1.00785},
    "XRP": {"ticker": "XRP-USD", "qty": 80.908713},
    "TRX": {"ticker": "TRX-USD", "qty": 34.790716},
}


# ----------------------------------------------------------------------------
# Prix de revient (PRU) : données personnelles, jamais dans le code ni dans git.
# Sources, par ordre de priorité : portfolio.json (local, ignoré par git) -> variable d'environnement
# PORTFOLIO_JSON (GitHub Actions) -> st.secrets["portfolio"] (Streamlit Cloud).
# Format : {"MSCI Europe": {"pru": 34.28}, "ETH": {"pru": 2795.2}, ...}
# ----------------------------------------------------------------------------
PORTFOLIO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio.json")


def load_costs() -> dict:
    try:
        with open(PORTFOLIO_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        pass
    raw = os.environ.get("PORTFOLIO_JSON")
    if raw:
        try:
            return json.loads(raw)
        except ValueError:
            log.warning("PORTFOLIO_JSON invalide : ignoré")
    try:
        import streamlit as st
        return {k: dict(v) for k, v in st.secrets["portfolio"].items()}
    except Exception:
        return {}


def save_costs(costs: dict) -> None:
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(costs, f, indent=2, ensure_ascii=False)


def apply_costs(costs: dict | None = None) -> None:
    """Injecte les PRU dans POSITIONS (champ « pru » lu par decision_table)."""
    costs = load_costs() if costs is None else costs
    for line, pos in POSITIONS.items():
        pru = (costs.get(line) or {}).get("pru")
        if pru and float(pru) > 0:
            pos["pru"] = float(pru)
        else:
            pos.pop("pru", None)


apply_costs()

COINGECKO_IDS = {
    "ETH-USD": "ethereum",
    "XRP-USD": "ripple",
    "TRX-USD": "tron",
}

DCA_DEFAULT_TARGETS = {
    "MSCI Europe": 35,
    "Nasdaq 100": 25,
    "MSCI Emerging Markets": 15,
    "Veolia": 5,
    "ETH": 15,
    "XRP": 2.5,
    "TRX": 2.5,
}

RSI_THRESHOLD = 35
LOOKBACK = 20
VOLUME_MULT = 2.0

# Indicateurs avancés
ATR_PERIOD = 14      # période de l'ATR (lissage de Wilder)
ATR_MULT = 2.0       # Stop-Loss = Prix - ATR_MULT x ATR
BB_PERIOD = 20       # Bandes de Bollinger
BB_STD = 2.0
OBV_LOOKBACK = 20    # fenêtre de comparaison prix / OBV

# ----------------------------------------------------------------------------
# Configuration Resend (Hardcodée)
# ----------------------------------------------------------------------------
def _get_secret(name: str) -> str:
    """Variable d'environnement d'abord, sinon st.secrets (import paresseux : Streamlit reste optionnel)."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st
        return str(st.secrets[name])
    except Exception:
        return ""


RESEND_API_KEY = _get_secret("RESEND_API_KEY")
DEST_EMAIL = "wolff.g@sfeir.com"
HISTORY_FILE = "history.csv"


# ----------------------------------------------------------------------------
# Données
# ----------------------------------------------------------------------------
@cached(ttl=600)
def fetch_yf(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period="4mo", interval="1d", auto_adjust=True)
    df = df[["Close", "High", "Volume"]].dropna().tail(60)
    df.index = df.index.tz_localize(None)
    return df


@cached(ttl=1800)
def fetch_coingecko(coin_id: str) -> pd.DataFrame:
    r = requests.get(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
        params={"vs_currency": "eur", "days": 60, "interval": "daily"},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    prices = pd.Series({pd.to_datetime(t, unit="ms"): p for t, p in data["prices"]})
    vols = pd.Series({pd.to_datetime(t, unit="ms"): v for t, v in data["total_volumes"]})
    df = pd.DataFrame({"Close": prices, "Volume": vols}).dropna()
    df["High"] = df["Close"]
    return df.tail(60)


def fetch_crypto(symbol: str) -> pd.DataFrame:
    coin_id, yf_ticker = WATCHLIST_CRYPTO[symbol]
    try:
        return fetch_coingecko(coin_id)
    except Exception:
        return fetch_yf(yf_ticker)


@cached(ttl=600)
def latest_price(ticker: str) -> float:
    try:
        return float(fetch_yf(ticker)["Close"].iloc[-1])
    except Exception:
        return 0.0


def eur_usd_rate() -> float:
    rate = latest_price("EURUSD=X")
    return rate if rate > 0 else 1.0


@cached(ttl=600)
def fetch_crypto_prices_eur() -> dict:
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": ",".join(COINGECKO_IDS.values()), "vs_currencies": "eur"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        return {tk: float(data[cg]["eur"]) for tk, cg in COINGECKO_IDS.items() if cg in data}
    except Exception:
        return {}


def unit_price_eur(line: str) -> float:
    pos = POSITIONS.get(line)
    if not pos:
        return 0.0
    ticker = pos["ticker"]
    if line in CRYPTO_LINES:
        eur_prices = fetch_crypto_prices_eur()
        if ticker in eur_prices:
            return eur_prices[ticker]
        return latest_price(ticker) / eur_usd_rate()
    return latest_price(ticker)


def position_value(line: str) -> float:
    pos = POSITIONS.get(line)
    if not pos:
        return 0.0
    return pos["qty"] * unit_price_eur(line)


# ----------------------------------------------------------------------------
# Analyse technique & Prix Cibles (Évolution 5)
# ----------------------------------------------------------------------------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def calculate_target_price_rsi30(df: pd.DataFrame, target_rsi: float = 30.0) -> float:
    """Calcule le prix théorique auquel l'actif atteindrait un RSI cible (30)."""
    if len(df) < 15:
        return 0.0
    close = df["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    
    last_gain = gain.iloc[-1]
    last_loss = loss.iloc[-1]
    last_close = close.iloc[-1]
    
    # Résolution de l'équation du Wilder's Smoothing RSI
    target_rs = target_rsi / (100 - target_rsi)
    required_change = (target_rs * (13 * last_loss) - 13 * last_gain) / (1 - target_rs)
    target_price = last_close + required_change
    return float(max(target_price, 0.0))


# ----------------------------------------------------------------------------
# Indicateurs avancés : OBV, ATR, Bollinger, MACD, Supports/Résistances
# ----------------------------------------------------------------------------
def obv_series(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume : cumul du volume signé par la direction du prix."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


OBV_LABELS = {
    "confirm_up": "🟢 Confirmé",
    "diverg_down": "🟠 Divergence ↘",
    "accumulation": "🟡 Accumulation",
    "confirm_down": "🔴 Faible",
}
OBV_HELP = {
    "🟢 Confirmé": "Prix et OBV montent ensemble : la hausse est soutenue par les volumes.",
    "🟠 Divergence ↘": "Le prix monte mais l'OBV baisse : hausse peu soutenue, risque d'essoufflement.",
    "🟡 Accumulation": "Le prix baisse mais l'OBV monte : les volumes entrants suggèrent une accumulation.",
    "🔴 Faible": "Prix et OBV baissent ensemble : la baisse est confirmée par les volumes.",
    "—": "Pas assez de données.",
}


def obv_signal(close: pd.Series, obv: pd.Series, lookback: int = OBV_LOOKBACK) -> str:
    if len(close) <= lookback:
        return "—"
    price_chg = float(close.iloc[-1] - close.iloc[-1 - lookback])
    obv_chg = float(obv.iloc[-1] - obv.iloc[-1 - lookback])
    if price_chg >= 0:
        return OBV_LABELS["confirm_up"] if obv_chg >= 0 else OBV_LABELS["diverg_down"]
    return OBV_LABELS["accumulation"] if obv_chg >= 0 else OBV_LABELS["confirm_down"]


def macd_state(hist_now: float, hist_prev: float) -> str:
    """Lecture de l'histogramme MACD : sa pente anticipe le croisement MACD/Signal."""
    if hist_now != hist_now or hist_prev != hist_prev:
        return "—"
    if hist_now >= 0:
        return "🟢 Haussier" if hist_now >= hist_prev else "🟠 Essoufflement"
    return "🟡 Rebond possible" if hist_now > hist_prev else "🔴 Baissier"


MACD_HELP = {
    "🟢 Haussier": "Histogramme positif et croissant : le momentum haussier s'accélère.",
    "🟠 Essoufflement": "Histogramme positif mais décroissant : la hausse perd de la vitesse (retournement possible).",
    "🟡 Rebond possible": "Histogramme négatif mais remontant : la baisse s'essouffle, retournement haussier possible.",
    "🔴 Baissier": "Histogramme négatif et décroissant : le momentum baissier s'accélère.",
    "—": "Pas assez de données.",
}


def atr_series(df: pd.DataFrame, n: int = ATR_PERIOD) -> pd.Series:
    """Average True Range (lissage de Wilder)."""
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def bollinger(close: pd.Series, n: int = BB_PERIOD, k: float = BB_STD):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd


def bollinger_label(pct_b: float) -> str:
    if pct_b != pct_b:
        return "—"
    if pct_b > 1:
        return "🔴 Hors bande haute"
    if pct_b > 0.8:
        return "🟠 Près bande haute"
    if pct_b < 0:
        return "🟢 Hors bande basse"
    if pct_b < 0.2:
        return "🟡 Près bande basse"
    return "⚪ Zone médiane"


def pivot_levels(df: pd.DataFrame, lookback: int = 20) -> dict:
    """Points pivots classiques calculés sur les `lookback` séances précédentes (≈ pivots mensuels)."""
    ref = df.iloc[-1 - lookback:-1]
    if ref.empty:
        return {}
    high, low, close = float(ref["High"].max()), float(ref["Low"].min()), float(ref["Close"].iloc[-1])
    p = (high + low + close) / 3
    return {
        "R2": p + (high - low), "R1": 2 * p - low, "Pivot": p,
        "S1": 2 * p - high, "S2": p - (high - low),
    }


def fib_levels(df: pd.DataFrame, lookback: int = 60) -> dict:
    """Retracements de Fibonacci sur le dernier mouvement (swing haut / swing bas)."""
    ref = df.iloc[-lookback:]
    if ref.empty:
        return {}
    high, low = float(ref["High"].max()), float(ref["Low"].min())
    rng = high - low
    ratios = {"23.6%": 0.236, "38.2%": 0.382, "50%": 0.5, "61.8%": 0.618, "78.6%": 0.786}
    uptrend = ref["High"].idxmax() > ref["Low"].idxmin()   # le plus haut est venu après le plus bas
    levels = {"Plus haut": high, "Plus bas": low}
    for name, r in ratios.items():
        levels[f"Fib {name}"] = high - r * rng if uptrend else low + r * rng
    return levels


def nearest_levels(levels: dict, price: float):
    """Support le plus proche sous le prix, résistance la plus proche au-dessus."""
    below = {k: v for k, v in levels.items() if v < price}
    above = {k: v for k, v in levels.items() if v > price}
    sup = max(below.items(), key=lambda kv: kv[1]) if below else None
    res = min(above.items(), key=lambda kv: kv[1]) if above else None
    return sup, res


def add_advanced_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute OBV, ATR, Bollinger, MACD et moyennes mobiles à un DataFrame OHLCV."""
    out = df.copy()
    out["OBV"] = obv_series(out["Close"], out["Volume"])
    out["ATR"] = atr_series(out)
    out["BB_mid"], out["BB_up"], out["BB_low"] = bollinger(out["Close"])
    out["EMA_12"] = out["Close"].ewm(span=12, adjust=False).mean()
    out["EMA_26"] = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = out["EMA_12"] - out["EMA_26"]
    out["Signal_Line"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["Signal_Line"]
    out["MM50"] = out["Close"].rolling(50).mean()
    out["MM200"] = out["Close"].rolling(200).mean()
    return out


@cached(ttl=1800)
def fetch_ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame:
    """OHLCV journalier complet (Open/High/Low/Close/Volume).

    Les cryptos cotées en USD sont converties en EUR avec le taux EUR/USD historique,
    pour rester cohérent avec le reste de l'application.
    """
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    if ticker.endswith("-USD"):
        fx = yf.Ticker("EURUSD=X").history(period=period, interval="1d")["Close"].dropna()
        if len(fx):
            fx.index = pd.DatetimeIndex(fx.index).tz_localize(None).normalize()
            fx = fx[~fx.index.duplicated(keep="last")]
            rate = fx.reindex(fx.index.union(df.index)).ffill().bfill().reindex(df.index)
        else:
            rate = pd.Series(eur_usd_rate(), index=df.index)
        df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].div(rate.to_numpy(), axis=0)
    return df


# ----------------------------------------------------------------------------
# Contexte macro : Fear & Greed (crypto) et prochains résultats (actions)
# ----------------------------------------------------------------------------
FNG_FR = {
    "Extreme Fear": "Peur extrême", "Fear": "Peur", "Neutral": "Neutre",
    "Greed": "Avidité", "Extreme Greed": "Avidité extrême",
}


@cached(ttl=3600)
def fetch_fear_greed() -> dict | None:
    """Crypto Fear & Greed Index (alternative.me), 0 = peur extrême, 100 = avidité extrême."""
    try:
        r = requests.get("https://api.alternative.me/fng/", params={"limit": 8}, timeout=10)
        r.raise_for_status()
        data = r.json()["data"]
        return {
            "value": int(data[0]["value"]),
            "label": FNG_FR.get(data[0]["value_classification"], data[0]["value_classification"]),
            "yesterday": int(data[1]["value"]) if len(data) > 1 else None,
            "week_ago": int(data[-1]["value"]) if len(data) > 1 else None,
        }
    except Exception:
        return None


@cached(ttl=6 * 3600)
def next_earnings(ticker: str):
    """Date des prochains résultats (None pour les ETF, cryptos ou si indisponible)."""
    try:
        cal = yf.Ticker(ticker).calendar
        if isinstance(cal, dict):
            raw = cal.get("Earnings Date", []) or []
        elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
            raw = list(cal.loc["Earnings Date"].dropna())
        else:
            raw = []
        today = datetime.date.today()
        future = sorted(d for d in (pd.to_datetime(x).date() for x in raw) if d >= today)
        return future[0] if future else None
    except Exception:
        return None


def analyse(df: pd.DataFrame) -> dict | None:
    if len(df) < LOOKBACK + 2:
        return None
        
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    last = df.iloc[-1]
    prev = df.iloc[-1 - LOOKBACK:-1]
    rsi_now = float(rsi(df["Close"]).iloc[-1])
    vol_ratio = float(last["Volume"] / prev["Volume"].mean()) if prev["Volume"].mean() > 0 else 0.0
    high20 = float(prev["High"].max())
    
    target_price_rsi30 = calculate_target_price_rsi30(df, 30.0)
    obv = obv_series(df["Close"], df["Volume"])
    
    return {
        "price": float(last["Close"]),
        "rsi": rsi_now,
        "high20": high20,
        "vol_ratio": vol_ratio,
        "perf_60d": float(last["Close"] / df["Close"].iloc[0] - 1),
        "dip": rsi_now < RSI_THRESHOLD,
        "breakout": bool(last["Close"] > high20 and vol_ratio > VOLUME_MULT),
        "macd": float(last["MACD"]),
        "macd_signal": float(last["Signal_Line"]),
        "macd_hist": float(last["MACD_Hist"]),
        "target_rsi30": target_price_rsi30,
        "obv_signal": obv_signal(df["Close"], obv),
        "macd_state": macd_state(float(last["MACD_Hist"]), float(df["MACD_Hist"].iloc[-2])),
    }


def run_scan() -> tuple[pd.DataFrame, dict]:
    rows, histories = [], {}
    jobs = [(t, grp, "Action", fetch_yf) for grp, ts in WATCHLIST_STOCKS.items() for t in ts]
    jobs += [(s, "Crypto", "Crypto", fetch_crypto) for s in WATCHLIST_CRYPTO]
    for symbol, group, kind, fetcher in jobs:
        try:
            df = fetcher(symbol)
            res = analyse(df)
        except Exception as e:
            log.warning("Scan : %s ignoré (%s)", symbol, e)
            res = None
        if res is None:
            log.warning("Scan : %s ignoré (données insuffisantes)", symbol)
            continue
        histories[symbol] = df
        rows.append({"Actif": symbol, "Univers": group, "Type": kind, **res})
    return pd.DataFrame(rows), histories


# ----------------------------------------------------------------------------
# Suivi Historique de la Valeur (Évolution 4)
# ----------------------------------------------------------------------------
def save_portfolio_snapshot(total_pea: float, total_crypto: float):
    """Sauvegarde l'historique de la valeur du portefeuille."""
    today = datetime.date.today().strftime("%Y-%m-%d")
    total_val = total_pea + total_crypto
    new_entry = pd.DataFrame([{"Date": today, "PEA": total_pea, "Crypto": total_crypto, "Total": total_val}])
    
    if os.path.exists(HISTORY_FILE):
        hist_df = pd.read_csv(HISTORY_FILE)
        hist_df = pd.concat([hist_df[hist_df["Date"] != today], new_entry], ignore_index=True)
    else:
        hist_df = new_entry
        
    hist_df.to_csv(HISTORY_FILE, index=False)
    return hist_df


def load_portfolio_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["Date", "PEA", "Crypto", "Total"])


# ----------------------------------------------------------------------------
# Aide à la décision
# ----------------------------------------------------------------------------
@cached(ttl=1800)
def fetch_close(ticker: str, period: str = "2y") -> pd.Series:
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    close = df["Close"].dropna()
    close.index = close.index.tz_localize(None)
    return close


def rsi_advice(value: float) -> str:
    if value < 30: return "🟢 Opportunité (Survente)"
    if value > 70: return "🔴 Attention (Surachat)"
    return "⚪ Neutre"


def trend_signal(price: float, mm200: float) -> str:
    if mm200 != mm200: return "—"
    return "🟢 Tendance Haussière" if price > mm200 else "🔴 Tendance Baissière"


@cached(ttl=1800)
def decision_table() -> pd.DataFrame:
    rows = []
    for line, pos in POSITIONS.items():
        try:
            close = fetch_close(pos["ticker"])
            price = float(close.iloc[-1])
            rsi_now = float(rsi(close).iloc[-1])
            mm50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else float("nan")
            mm200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float("nan")
            pru = pos.get("pru")
            if pru and pru > 0:
                unit = unit_price_eur(line)
                pv = unit / pru - 1
                pv_eur = pos["qty"] * (unit - pru)
                invested = pos["qty"] * pru
            else:
                pv = price / float(close.iloc[0]) - 1 if len(close) else float("nan")   # estimation sur 2 ans
                pv_eur = invested = float("nan")
                pru = float("nan")
        except Exception as e:
            log.warning("Aide à la décision : %s (%s) ignoré (%s)", line, pos["ticker"], e)
            continue
        rows.append({
            "Univers": "Crypto" if line in CRYPTO_LINES else "PEA",
            "Ligne": line, "Ticker": pos["ticker"], "Prix": price,
            "RSI(14)": rsi_now, "Conseil RSI": rsi_advice(rsi_now),
            "MM50": mm50, "MM200": mm200, "Tendance": trend_signal(price, mm200),
            "PV latente": pv, "PRU": pru, "PV (€)": pv_eur, "Investi (€)": invested,
            "PV réelle": invested == invested,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Envoi d'e-mails (Resend API)
# ----------------------------------------------------------------------------
def send_resend_email(subject, body):
    resend.api_key = RESEND_API_KEY
    try:
        resend.Emails.send({
            "from": "BOURSE APP <onboarding@resend.dev>",
            "to": DEST_EMAIL,
            "subject": subject,
            "text": body,
        })
        return True, f"✅ E-mail envoyé avec succès à {DEST_EMAIL} !"
    except Exception as e:
        return False, f"❌ Erreur Resend : {e}"

ALERT_RSI_LOW = 35    # opportunité
ALERT_RSI_HIGH = 75   # surachat


def rsi_alerts(deci: pd.DataFrame) -> list:
    msgs = []
    if deci is None or deci.empty: return msgs
    for _, r in deci.iterrows():
        val = r["RSI(14)"]
        if val < ALERT_RSI_LOW: msgs.append(f"🟢 {r['Ligne']} ({r['Ticker']}) — Opportunité : RSI {val:.0f} (< 35)")
        elif val > ALERT_RSI_HIGH: msgs.append(f"🔴 {r['Ligne']} ({r['Ticker']}) — Surachat : RSI {val:.0f} (> 75)")
    return msgs


# ----------------------------------------------------------------------------
# Analyse d'un signal (utilisée par l'e-mail quotidien)
# ----------------------------------------------------------------------------
# Symboles Yahoo erronés (cotation figée, volume nul) -> symbole valide, pour l'analyse approfondie.
YF_ANALYSIS_TICKERS = {"ARB-USD": "ARB11841-USD"}

CONFIRMING_OBV = (OBV_LABELS["confirm_up"], OBV_LABELS["accumulation"])
CONFIRMING_MACD = ("🟢 Haussier", "🟡 Rebond possible")


def signal_snapshot(ticker: str, atr_mult: float = ATR_MULT) -> dict:
    """Prix, ATR/Stop-Loss, OBV, MACD, support/résistance (pivots) et prix cible RSI 30 d'un actif.

    Lève ValueError si l'historique est insuffisant ou si les données semblent figées.
    """
    d = add_advanced_columns(fetch_ohlcv(ticker))
    if len(d) < 60:
        raise ValueError("historique insuffisant")
    if d["Volume"].tail(10).median() == 0:
        raise ValueError("données figées (volume nul)")
    last, prev = d.iloc[-1], d.iloc[-2]
    price, atr = float(last["Close"]), float(last["ATR"])
    support, resistance = nearest_levels(pivot_levels(d), price)
    obv_lbl = obv_signal(d["Close"], d["OBV"])
    macd_lbl = macd_state(float(last["MACD_Hist"]), float(prev["MACD_Hist"]))
    return {
        "price": price,
        "atr": atr,
        "stop": price - atr_mult * atr,
        "stop_pct": -atr_mult * atr / price,
        "obv": obv_lbl,
        "macd": macd_lbl,
        "support": support,
        "resistance": resistance,
        "target_rsi30": calculate_target_price_rsi30(d),
        "dip_confirmed": obv_lbl in CONFIRMING_OBV or macd_lbl in CONFIRMING_MACD,
    }


# ----------------------------------------------------------------------------
# Taille de position selon le risque
# ----------------------------------------------------------------------------
SIZING_DEFAULTS = {"risk_pct": 1.0, "cap_pct": 20.0, "budget": 200.0}   # budget = enveloppe par signal (e-mail)


def position_size(capital: float, risk_pct: float, price: float, stop: float, cap_pct: float,
                  available: float, integer: bool, current_value: float = 0.0) -> dict:
    """Quantité = (capital × risque %) / (prix − stop), plafonnée par le plafond de la ligne et l'enveloppe.

    - plafond de la ligne : cap_pct % du capital, moins la valeur déjà détenue sur cette ligne ;
    - enveloppe : montant disponible à investir (budget) ;
    - integer=True (actions/ETF) : quantité arrondie à l'entier inférieur ; False (crypto) : fractions permises.
    Retourne qty, amount, risk_eur, limited_by (None / "plafond par ligne" / "enveloppe disponible"), ok, reason.
    """
    res = {"qty": 0.0, "amount": 0.0, "risk_eur": 0.0, "limited_by": None, "ok": False, "reason": ""}

    def bad(msg):
        res["reason"] = msg
        return res

    vals = (capital, risk_pct, price, stop, cap_pct, available, current_value)
    if any(v is None or (isinstance(v, float) and not math.isfinite(v)) for v in vals):
        return bad("donnée manquante (ATR ou prix indisponible)")
    if price <= 0:
        return bad("prix invalide")
    if capital <= 0:
        return bad("capital nul")
    if risk_pct <= 0:
        return bad("risque par trade nul")
    if stop >= price:
        return bad("stop au-dessus du prix (ATR nul ou invalide)")
    risk_per_unit = price - stop
    qty = capital * risk_pct / 100 / risk_per_unit
    room_line = max(cap_pct / 100 * capital - current_value, 0.0)
    avail = max(available, 0.0)
    max_amount = min(room_line, avail)
    if qty * price > max_amount:
        res["limited_by"] = "plafond par ligne" if room_line < avail else "enveloppe disponible"
        qty = max_amount / price
    if integer:
        qty = math.floor(qty + 1e-9)
    if qty <= 0:
        return bad("aucune marge disponible" if max_amount <= 0 else "moins d'un titre dans la limite autorisée")
    res.update(qty=float(qty), amount=float(qty * price), risk_eur=float(qty * risk_per_unit), ok=True)
    return res


def eur_factor(ticker: str, kind: str) -> float:
    """Diviseur pour convertir le prix natif en EUR : le taux EUR/USD pour les actions américaines, 1 sinon
    (ETF/actions françaises en EUR ; cryptos déjà converties par fetch_ohlcv)."""
    if kind == "Action" and not ticker.endswith(".PA"):
        return eur_usd_rate()
    return 1.0


def portfolio_value() -> float:
    """Valeur totale du portefeuille (€) : somme des lignes de POSITIONS."""
    return float(sum(position_value(line) for line in POSITIONS))


def line_for_ticker(ticker: str):
    for line, pos in POSITIONS.items():
        if pos["ticker"] == ticker:
            return line
    return None


def size_signal(snap: dict, ticker: str, kind: str, capital: float, risk_pct: float, cap_pct: float,
                available: float) -> dict:
    """position_size appliquée à un instantané (prix, stop) ; conversion en EUR et valeur déjà détenue incluses."""
    fx = eur_factor(ticker, kind)
    line = line_for_ticker(ticker)
    held = position_value(line) if line else 0.0
    res = position_size(capital, risk_pct, snap["price"] / fx, snap["stop"] / fx, cap_pct, available,
                        integer=(kind != "Crypto"), current_value=held)
    res["price_eur"] = snap["price"] / fx
    return res
