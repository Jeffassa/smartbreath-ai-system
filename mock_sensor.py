import requests
import time
import random
import os
import psycopg2
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
URL_API = os.getenv("API_URL", "http://127.0.0.1:8000/analyze")

def get_patient_by_email(email):
    """Récupère les informations complètes du patient depuis la base de données"""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "esante_respiratoire"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD")
        )
        cur = conn.cursor()
        query = """
            SELECT patient_id, nom, prenom, date_naissance, taille_cm, pathologie, est_fumeur 
            FROM patients WHERE email = %s
        """
        cur.execute(query, (email,))
        res = cur.fetchone()
        cur.close(); conn.close()
        
        if res:
            dob = res[3]
            age = (datetime.now().year - dob.year) if dob else 45
            return {
                "id": str(res[0]), "nom": res[1], "prenom": res[2],
                "age": age, "height": res[4] or 170, 
                "pathologie": res[5] or "Non spécifié", "est_fumeur": bool(res[6])
            }
    except Exception as e:
        print(f"Erreur lecture BDD : {e}"); return None

class PhysiologicalSimulator:
    def __init__(self, patient_info):
        self.patient = patient_info
        self.step = 0
        self.buffer = [] # Pour calcul des tendances (spo2, bpm)
        
        # Initialisation des baselines selon le profil
        age_factor = (self.patient['age'] - 40) * 0.05
        smoker_penalty = 2.0 if self.patient['est_fumeur'] else 0.0
        
        self.baseline_spo2 = 98.0 - age_factor - smoker_penalty
        self.baseline_bpm = 70 + (self.patient['age'] - 40) * 0.2 + (5 if self.patient['est_fumeur'] else 0)
        self.baseline_temp = 36.6
        self.baseline_muscle = 75.0 - age_factor * 2
        self.baseline_flow = 4.2 - age_factor * 0.05
        
        # État actuel
        self.current_spo2 = self.baseline_spo2
        self.current_bpm = self.baseline_bpm
        self.current_temp = self.baseline_temp
        self.current_muscle = self.baseline_muscle
        self.current_flow = self.baseline_flow
        
        self.scenarios = [
            {"name": "Crise Rapide", "start": 25, "end": 45, "severity": 1.2},
            {"name": "Crise Lente", "start": 30, "end": 60, "severity": 0.8},
            {"name": "Crise Sévère", "start": 20, "end": 40, "severity": 1.5},
            {"name": "Stable Prolongé", "start": 999, "end": 999, "severity": 0.0},
        ]
        self.current_scenario = random.choice(self.scenarios[:3]) if random.random() < 0.7 else self.scenarios[3]

    def get_phase(self):
        cycle_pos = self.step % 100
        scenario = self.current_scenario
        if scenario['start'] <= cycle_pos < scenario['end']:
            progress = (cycle_pos - scenario['start']) / (scenario['end'] - scenario['start'])
            if progress < 0.6:
                decay = np.exp(-5 * (1 - progress / 0.6))
                return "DÉGRADATION", decay * scenario['severity']
            return "CRISE AIGUË", scenario['severity']
        elif scenario['end'] <= cycle_pos < scenario['end'] + 20:
            recovery = (cycle_pos - scenario['end']) / 20
            return "RÉCUPÉRATION", (1 - recovery) * scenario['severity'] * 0.5
        return "STABLE", 0.0

    def generate_measure(self):
        phase, intensity = self.get_phase()
        
        # Calcul des cibles avec intégration de la température
        if phase == "STABLE":
            target_spo2, target_bpm = self.baseline_spo2, self.baseline_bpm
            target_temp = self.baseline_temp + random.uniform(-0.1, 0.1)
        elif phase == "DÉGRADATION" or phase == "CRISE AIGUË":
            target_spo2 = self.baseline_spo2 - (15 * intensity)
            target_bpm = self.baseline_bpm + (50 * intensity)
            # La température monte si infection ou crise sévère
            target_temp = self.baseline_temp + (2.5 * intensity)
        else: # RÉCUPÉRATION
            target_spo2 = self.baseline_spo2 - (5 * intensity)
            target_bpm = self.baseline_bpm + (15 * intensity)
            target_temp = self.baseline_temp + (0.5 * intensity)

        # Lissage (Inertie physiologique)
        smoothing = 0.15
        self.current_spo2 += (target_spo2 - self.current_spo2) * smoothing
        self.current_bpm += (target_bpm - self.current_bpm) * smoothing
        self.current_temp += (target_temp - self.current_temp) * 0.1 # Plus lent pour la temp
        
        final_spo2 = round(max(70.0, min(100.0, self.current_spo2 + random.uniform(-0.3, 0.3))), 1)
        final_bpm = int(max(40, min(160, self.current_bpm + random.uniform(-1, 1))))
        final_temp = round(self.current_temp + random.uniform(-0.05, 0.05), 1)

        # Calcul des tendances et volatilité (Buffer de 5 mesures)
        self.buffer.append({'spo2': final_spo2, 'bpm': final_bpm})
        if len(self.buffer) > 5: self.buffer.pop(0)

        spo2_trend = self.buffer[-1]['spo2'] - self.buffer[0]['spo2'] if len(self.buffer) > 1 else 0
        bpm_trend = self.buffer[-1]['bpm'] - self.buffer[0]['bpm'] if len(self.buffer) > 1 else 0
        spo2_vol = np.std([x['spo2'] for x in self.buffer]) if len(self.buffer) > 1 else 0

        return {
            "patient_id": self.patient["id"],
            "spo2": final_spo2,
            "bpm": final_bpm,
            "temperature": final_temp,
            "muscle_strength": round(self.current_muscle, 1),
            "flow_rate": round(self.current_flow, 2),
            "age": self.patient['age'],
            "height": self.patient['height'],
            "pathologie_enc": 1, 
            "is_smoker": int(self.patient['est_fumeur']),
            "spo2_trend": round(spo2_trend, 2),
            "bpm_trend": round(bpm_trend, 2),
            "spo2_volatility": round(spo2_vol, 2),
            "phase": phase
        }

    def next_step(self):
        self.step += 1
        if self.step % 100 == 0:
            self.current_scenario = random.choice(self.scenarios[:3]) if random.random() < 0.7 else self.scenarios[3]

# --- BOUCLE PRINCIPALE ---
print("Démarrage du Simulateur Physiologique v2.5 (Température & IA Trends)")
email_input = input("Email du patient : ").strip()
patient = get_patient_by_email(email_input)

if not patient:
    print("Erreur : Patient introuvable.")
    exit(1)

simulator = PhysiologicalSimulator(patient)
print(f"Simulation lancée pour {patient['prenom']} {patient['nom']} (Scénario: {simulator.current_scenario['name']})")

try:
    while True:
        measure = simulator.generate_measure()
        phase = measure.pop('phase')
        
        try:
            response = requests.post(URL_API, json=measure, timeout=5)
            if response.status_code == 200:
                data = response.json()
                status = data.get('status', 'STABLE')
                risk = data.get('risk_score', 0) * 100
                
                # Formatage console
                color = "\033[92m" # Vert
                if status == "CRITIQUE": color = "\033[91m" # Rouge
                elif status == "PRÉVENTION": color = "\033[93m" # Jaune
                
                print(f"[{simulator.step:03d}] {phase:12s} | "
                      f"SpO2: {measure['spo2']}% ({measure['spo2_trend']:+g}) | "
                      f"Temp: {measure['temperature']}°C | "
                      f"Risque: {risk:4.1f}% | {color}{status}\033[0m")
            else:
                print(f"Erreur API: {response.status_code}")
        except Exception as e:
            print(f"Erreur connexion : {e}")
        
        simulator.next_step()
        time.sleep(1.5)
except KeyboardInterrupt:
    print("\nSimulation arrêtée par l'utilisateur.")