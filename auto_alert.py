"""Script d'exécution automatique quotidienne d'alertes."""
import resend
from core import run_scan, rsi_alerts, decision_table, RESEND_API_KEY, DEST_EMAIL

def check_and_send():
    deci = decision_table()
    alerts = rsi_alerts(deci)
    
    scan, _ = run_scan()
    signals = scan[scan["dip"]] if not scan.empty else scan
    
    for _, r in signals.iterrows():
        alerts.append(f"🔥 Signal Opportunité (Dip) sur {r['Actif']} — RSI: {r['rsi']:.1f}, Prix Cible: {r['target_rsi30']:.2f}€")
        
    if alerts:
        resend.api_key = RESEND_API_KEY
        body = "📊 Rapport d'opportunités automatiques :\n\n" + "\n".join(alerts)
        resend.Emails.send({
            "from": "BOURSE APP <onboarding@resend.dev>",
            "to": DEST_EMAIL,
            "subject": "⚡ Rapport Automatique d'Opportunités Bourse",
            "text": body,
        })
        print(f"Alerte envoyée à {DEST_EMAIL}")
    else:
        print("Aucun signal critique aujourd'hui.")

if __name__ == "__main__":
    check_and_send()
