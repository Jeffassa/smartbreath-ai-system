import requests
import time
import random
import os
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Configuration
URL_API = os.getenv("API_URL", "http://127.0.0.1:8000/analyze")
PATIENT_ID = os.getenv("TEST_PATIENT_ID") 

def generate_telemetry(step):
    """Genere des donnees normales ou simule une crise cyclique."""
    # Creation d'une crise toutes les 15 iterations (environ 30 secondes)
    # Les iterations 10 a 14 simulent la degradation
    is_crisis = (step % 15 >= 10) 

    if not is_crisis:
        # --- ETAT NORMAL ---
        spo2 = random.uniform(96.0, 99.0)
        bpm = random.randint(70, 85)
        flow_rate = random.uniform(3.5, 4.8)
        muscle_strength = random.uniform(70.0, 90.0)
    else:
        # --- ETAT DE CRISE (Degradation progressive) ---
        spo2 = random.uniform(88.0, 93.0)
        bpm = random.randint(100, 120)
        flow_rate = random.uniform(1.2, 2.2)
        muscle_strength = random.uniform(30.0, 50.0)

    # Le payload correspond strictement au modele Pydantic RespiratoryMeasure du Backend
    return {
        "patient_id": str(PATIENT_ID),
        "spo2": round(spo2, 1),
        "bpm": int(bpm),
        "flow_rate": round(flow_rate, 2),
        "muscle_strength": round(muscle_strength, 1)
    }

print(f"--- Simulateur demarre pour le patient : {PATIENT_ID} ---")
print(f"--- Envoi des donnees vers : {URL_API} ---")

step = 0
try:
    while True:
        payload = generate_telemetry(step)
        
        try:
            # Envoi a FastAPI
            response = requests.post(URL_API, json=payload, timeout=10)
            
            if response.status_code == 200:
                res_json = response.json()
                # On recupere 'condition' qui est le champ renvoye par ton Backend
                status = res_json.get('condition', 'Inconnu')
                ttc = res_json.get('ttc', 'N/A')
                
                # Gestion des couleurs pour le terminal (fonctionne sur la plupart des terminaux modernes)
                # Vert pour NORMAL, Rouge pour le reste
                color = "\033[92m" if status == "NORMAL" else "\033[91m"
                reset = "\033[0m"
                
                print(f"[{step}] {color}Statut: {status}{reset} | SpO2: {payload['spo2']}% | BPM: {payload['bpm']} | TTC: {ttc} min")
            else:
                print(f"Alerte Serveur: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("Erreur : Serveur backend injoignable. Verifiez qu'Uvicorn est lance.")
        except Exception as e:
            print(f"Erreur imprevue : {e}")
        
        step += 1
        time.sleep(2)

except KeyboardInterrupt:
    print("\nSimulateur arrete.")