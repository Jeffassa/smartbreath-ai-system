import xgboost as xgb
import numpy as np
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class RespiratoryAI:
    def __init__(self, model_path=None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # Test des deux chemins possibles (structure simple ou doublée)
            paths_to_check = [
                os.path.join(base_dir, 'ml_engine', 'models', 'respiratory_model.json'),
                os.path.join(base_dir, 'ml_engine', 'ml_engine', 'models', 'respiratory_model.json')
            ]
            for p in paths_to_check:
                if os.path.exists(p):
                    model_path = p
                    break
        
        self.model = xgb.Booster()
        self.feature_names = ['spo2', 'bpm', 'muscle_strength', 'flow_rate', 'age', 'height']
        
        try:
            if model_path and os.path.exists(model_path):
                self.model.load_model(model_path)
                logger.info(f"Modele charge : {model_path}")
            else:
                logger.error(f"Fichier modele introuvable")
        except Exception as e:
            logger.error(f"Erreur chargement modele : {e}")

    def predict(self, raw_data):
        try:
            features = [
                float(raw_data['spo2']),
                float(raw_data['bpm']),
                float(raw_data.get('muscle_strength', 50.0)),
                float(raw_data.get('flow_rate', 3.0)),
                int(raw_data['age']),
                float(raw_data['height'])
            ]
            
            dmatrix = xgb.DMatrix([features], feature_names=self.feature_names)
            score = float(self.model.predict(dmatrix)[0])
            
            age = int(raw_data['age'])
            bpm = float(raw_data['bpm'])
            flow = float(raw_data.get('flow_rate', 3.0))
            path_guess = ""

            if score > 0.5:
                if age > 50:
                    path_guess = " (Suspicion Asthme Age)" if bpm >= 105 else " (Suspicion BPCO)" if flow < 2.3 else ""
                elif flow < 2.5 or bpm > 100:
                    path_guess = " (Suspicion Crise Asthme)"

            status = "NORMAL"
            rec = "RAS - Continuez le suivi"
            
            if score > 0.8:
                status = "CRITIQUE"
                rec = f"Urgence medicale{path_guess}. Risque vital detecte."
            elif score > 0.5:
                status = "MODERE"
                rec = f"Surveillance rapprochee{path_guess}. Signes de detresse moderes."

            return {"status": status, "risk_score": score, "recommendation": rec}
            
        except Exception as e:
            logger.error(f"Erreur Inference : {e}")
            return {"status": "ERREUR IA", "risk_score": 0.0, "recommendation": "Donnees patient incompletes."}