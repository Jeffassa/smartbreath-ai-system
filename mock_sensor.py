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
URL_FEEDBACK = os.getenv("FEEDBACK_URL", "http://127.0.0.1:8000/feedback")

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
        self.buffer = [] 
        
        # Initialisation des baselines selon le profil
        age_factor = (self.patient['age'] - 40) * 0.05
        smoker_penalty = 2.0 if self.patient['est_fumeur'] else 0.0
        
        self.baseline_spo2 = 98.0 - age_factor - smoker_penalty
        self.baseline_bpm = 70 + (self.patient['age'] - 40) * 0.2 + (5 if self.patient['est_fumeur'] else 0)
        self.baseline_temp = 36.6
        self.baseline_muscle = 75.0 - age_factor * 2
        self.baseline_flow = 4.2 - age_factor * 0.05
        
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
        
        if phase == "STABLE":
            target_spo2, target_bpm = self.baseline_spo2, self.baseline_bpm
            target_temp = self.baseline_temp + random.uniform(-0.1, 0.1)
        elif phase == "DÉGRADATION" or phase == "CRISE AIGUË":
            target_spo2 = self.baseline_spo2 - (15 * intensity)
            target_bpm = self.baseline_bpm + (50 * intensity)
            target_temp = self.baseline_temp + (2.5 * intensity)
        else: # RÉCUPÉRATION
            target_spo2 = self.baseline_spo2 - (5 * intensity)
            target_bpm = self.baseline_bpm + (15 * intensity)
            target_temp = self.baseline_temp + (0.5 * intensity)

        smoothing = 0.15
        self.current_spo2 += (target_spo2 - self.current_spo2) * smoothing
        self.current_bpm += (target_bpm - self.current_bpm) * smoothing
        self.current_temp += (target_temp - self.current_temp) * 0.1 
        
        final_spo2 = round(max(70.0, min(100.0, self.current_spo2 + random.uniform(-0.3, 0.3))), 1)
        final_bpm = int(max(40, min(160, self.current_bpm + random.uniform(-1, 1))))
        final_temp = round(self.current_temp + random.uniform(-0.05, 0.05), 1)

        self.buffer.append({'spo2': final_spo2, 'bpm': final_bpm})
        if len(self.buffer) > 5: self.buffer.pop(0)

        return {
            "patient_id": self.patient["id"],
            "spo2": final_spo2,
            "bpm": final_bpm,
            "temperature": final_temp,
            "muscle_strength": round(self.current_muscle, 1),
            "flow_rate": round(self.current_flow, 2),
            "phase": phase
        }

    def next_step(self):
        self.step += 1
        if self.step % 100 == 0:
            self.current_scenario = random.choice(self.scenarios[:3]) if random.random() < 0.7 else self.scenarios[3]

# --- BOUCLE PRINCIPALE ---
print("Démarrage du Simulateur Physiologique v3.0 (Boucle de Feedback IA)")
email_input = input("Email du patient : ").strip()
patient = get_patient_by_email(email_input)

if not patient:
    print("Erreur : Patient introuvable."); exit(1)

simulator = PhysiologicalSimulator(patient)
print(f"Simulation lancée pour {patient['prenom']} {patient['nom']}")

try:
    while True:
        measure = simulator.generate_measure()
        phase = measure.pop('phase')
        
        try:
            # 1. ENVOI DE LA MESURE
            response = requests.post(URL_API, json=measure, timeout=5)
            if response.status_code == 200:
                data = response.json()
                data_id = data.get('data_id') # Très important pour le feedback !
                status = data.get('status', 'STABLE')
                risk = data.get('risk_score', 0)
                
                # 2. SIMULATION DU FEEDBACK PATIENT (Apprentissage supervisé)
                # Si le risque est > 60%, le patient confirme souvent qu'il va mal
                if risk > 0.6 and random.random() < 0.85:
                    feedback_payload = {
                        "data_id": data_id,
                        "actual_outcome": 1, # "Je me sens mal"
                        "comment": "Simulation: Patient confirme la gêne respiratoire."
                    }
                    requests.post(URL_FEEDBACK, json=feedback_payload)
                elif risk < 0.2 and random.random() < 0.1:
                    # Rarement, on envoie un feedback "Tout va bien" pour confirmer la stabilité
                    feedback_payload = {"data_id": data_id, "actual_outcome": 0, "comment": "Simulation: Patient confirme que tout va bien."}
                    requests.post(URL_FEEDBACK, json=feedback_payload)

                # Affichage console
                color = "\033[92m" if status == "STABLE" else "\033[93m" if status == "PRÉVENTION" else "\033[91m"
                print(f"[{simulator.step:03d}] {phase:12s} | SpO2: {measure['spo2']}% | Temp: {measure['temperature']}°C | Risque: {risk*100:4.1f}% | {color}{status}\033[0m")
            
        except Exception as e:
            print(f"Erreur : {e}")
        
        simulator.next_step()
        time.sleep(1.5)
except KeyboardInterrupt:
    print("\nSimulation arrêtée.")