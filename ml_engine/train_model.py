import xgboost as xgb
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def generate_time_series_data(n_patients=500, timesteps_per_patient=60):
    np.random.seed(42)
    all_data = []
    
    for patient_id in range(n_patients):
        age = np.random.randint(18, 90)
        is_smoker = np.random.choice([0, 1], p=[0.7, 0.3])
        pathologie = np.random.choice([0, 1, 2, 3, 4])
        height = np.random.randint(150, 200)
        will_have_crisis = np.random.rand() < 0.3
        crisis_start = np.random.randint(25, 45) if will_have_crisis else 100
        
        patient_records = []
        for t in range(timesteps_per_patient):
            time_to_crisis = max(0, crisis_start - t)
            
            # Base physiologique
            base_spo2 = 98.0 - (age - 40) * 0.05 - is_smoker * 2.0
            base_bpm = 70 + (age - 40) * 0.2 + is_smoker * 5
            
            if will_have_crisis and t >= crisis_start - 15:
                decay = np.exp(-(time_to_crisis / 6.0))
                spo2 = base_spo2 - (12 * decay) + np.random.normal(0, 0.5)
                bpm = base_bpm + (35 * decay) + np.random.normal(0, 2)
                # La température monte en cas de crise (infection/inflammation)
                temp = 36.6 + (2.5 * decay) + np.random.normal(0, 0.2)
                muscle = 70 - (20 * decay) + np.random.normal(0, 3)
                flow = 4.0 - (1.5 * decay) + np.random.normal(0, 0.2)
            else:
                spo2 = base_spo2 + np.random.normal(0, 0.8)
                bpm = base_bpm + np.random.normal(0, 4)
                temp = 36.6 + np.random.normal(0, 0.15)
                muscle = np.random.uniform(65, 85)
                flow = np.random.uniform(3.7, 4.4)

            patient_records.append({
                'patient_id': patient_id, 'spo2': max(70, min(100, spo2)),
                'bpm': max(40, min(160, bpm)), 'temperature': round(temp, 1),
                'muscle_strength': muscle, 'flow_rate': flow,
                'age': age, 'height': height, 'is_smoker': is_smoker, 'pathologie_enc': pathologie,
                'target': 1 if (will_have_crisis and 0 <= time_to_crisis <= 7) else 0
            })
        
        pdf = pd.DataFrame(patient_records)
        pdf['spo2_trend'] = pdf['spo2'].diff(periods=3).fillna(0)
        pdf['bpm_trend'] = pdf['bpm'].diff(periods=3).fillna(0)
        pdf['spo2_volatility'] = pdf['spo2'].rolling(window=3).std().fillna(0)
        all_data.append(pdf)

    return pd.concat(all_data, ignore_index=True)

# Configuration et Entraînement
df = generate_time_series_data()
features = ['spo2', 'bpm', 'temperature', 'muscle_strength', 'flow_rate', 'age', 
            'height', 'pathologie_enc', 'is_smoker', 'spo2_trend', 'bpm_trend', 'spo2_volatility']

X = df[features]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, scale_pos_weight=3.5)
model.fit(X_train, y_train)

booster = model.get_booster()
booster.save_model('ml_engine/models/respiratory_model_predictive.json')

print("Modèle IA entraîné et sauvegardé avec succès dans ml_engine/models/")