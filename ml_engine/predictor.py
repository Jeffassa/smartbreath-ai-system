import xgboost as xgb
import pandas as pd
import numpy as np
import os
from collections import deque
import logging

logger = logging.getLogger(__name__)

class RespiratoryAI:
    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, "models", "respiratory_model_predictive.json")
        
        if not os.path.exists(model_path):
            error_msg = f" Modèle XGBoost introuvable à : {model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            self.model = xgb.Booster()
            self.model.load_model(model_path)
            self.history = {} 
            self.model_ready = True
            print(f" IA SmartBreath chargée avec succès depuis : {model_path}")
        except Exception as e:
            logger.error(f" Erreur lors du chargement du modèle : {e}")
            raise e

    def predict(self, data):
        """
        Prend en entrée les données capteurs et le contexte patient,
        calcule les tendances et renvoie un score de risque et un statut.
        """
        p_id = str(data.get('patient_id', 'unknown'))
        
        if p_id not in self.history:
            self.history[p_id] = deque(maxlen=5)
        
        self.history[p_id].append({
            'spo2': data.get('spo2', 95), 
            'bpm': data.get('bpm', 70)
        })

        hist = list(self.history[p_id])
        spo2_trend = hist[-1]['spo2'] - hist[0]['spo2'] if len(hist) > 1 else 0
        bpm_trend = hist[-1]['bpm'] - hist[0]['bpm'] if len(hist) > 1 else 0
        spo2_vol = np.std([x['spo2'] for x in hist]) if len(hist) > 1 else 0

        feat_cols = [
            'spo2', 'bpm', 'temperature', 'muscle_strength', 'flow_rate', 'age', 
            'height', 'pathologie_enc', 'is_smoker', 'spo2_trend', 'bpm_trend', 'spo2_volatility'
        ]
        
        feat_values = pd.DataFrame([{
            'spo2': data.get('spo2', 95),
            'bpm': data.get('bpm', 70),
            'temperature': data.get('temperature', 36.6),
            'muscle_strength': data.get('muscle_strength', 75.0),
            'flow_rate': data.get('flow_rate', 4.0),
            'age': data.get('age', 45),
            'height': data.get('height', 170),
            'pathologie_enc': data.get('pathologie_enc', 1), 
            'is_smoker': int(data.get('is_smoker', False)),
            'spo2_trend': spo2_trend,
            'bpm_trend': bpm_trend,
            'spo2_volatility': spo2_vol
        }])

        dmatrix = xgb.DMatrix(feat_values[feat_cols])
        proba = float(self.model.predict(dmatrix)[0])
        
        temp = data.get('temperature', 36.6)
        spo2 = data.get('spo2', 95)

        if proba > 0.80 or spo2 < 88:
            status = "CRITIQUE"
            rec = "Alerte : Insuffisance respiratoire sévère détectée. Contactez les urgences."
        elif proba > 0.60 or (temp > 38.5 and proba > 0.4):
            status = "PRÉVENTION"
            rec = "Risque élevé : Fièvre et instabilité respiratoire. Consultez un médecin rapidement."
        elif proba > 0.35 or spo2 < 93:
            status = "SURVEILLANCE"
            rec = "Vigilance : Signes de fatigue détectés. Reposez-vous et suivez votre traitement."
        else:
            status = "STABLE"
            rec = "Tout est normal. Continuez votre suivi habituel."

        return {
            "risk_score": proba, 
            "status": status, 
            "recommendation": rec
        }