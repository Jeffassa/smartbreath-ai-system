import xgboost as xgb
import pandas as pd
import numpy as np
import os
import psycopg2
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION BDD ---
def get_real_feedback_data():
    """Récupère les données réelles validées par les patients dans la BDD"""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        # On ne prend que les lignes où le patient a donné un feedback (actual_outcome)
        query = """
            SELECT s.spo2, s.bpm, s.temperature, s.muscle_strength, s.flow_rate, 
                   p.age, p.taille_cm as height, 1 as pathologie_enc, 
                   p.est_fumeur::int as is_smoker, s.actual_outcome as target
            FROM sensor_data s
            JOIN patients p ON s.patient_id = p.patient_id
            WHERE s.actual_outcome IS NOT NULL
        """
        df_real = pd.read_sql(query, conn)
        conn.close()
        
        if not df_real.empty:
            # Recalcul des tendances pour les données réelles
            df_real['spo2_trend'] = df_real['spo2'].diff().fillna(0)
            df_real['bpm_trend'] = df_real['bpm'].diff().fillna(0)
            df_real['spo2_volatility'] = df_real['spo2'].rolling(window=3).std().fillna(0)
            print(f"{len(df_real)} feedbacks réels récupérés pour l'entraînement.")
            return df_real
        return None
    except Exception as e:
        print(f"Impossible de lire les feedbacks réels (BDD vide ?) : {e}")
        return None

def generate_simulated_base_data(n_patients=300):
    """Génère la base théorique de données pour l'IA"""
    all_data = []
    for patient_id in range(n_patients):
        pass 
    return df_simulated # Simplifié pour l'exemple

# --- SCRIPT PRINCIPAL D'ENTRAÎNEMENT ---

print(" Démarrage de l'entraînement hybride (Théorie + Feedback Réel)...")

# 1. Charger les deux sources
df_sim = generate_time_series_data(n_patients=400) # Base théorique
df_real = get_real_feedback_data() # Expérience patient

# 2. Fusionner les données
if df_real is not None:
    df = pd.concat([df_sim, df_real], ignore_index=True)
else:
    df = df_sim

# 3. Préparation des features
features = ['spo2', 'bpm', 'temperature', 'muscle_strength', 'flow_rate', 'age', 
            'height', 'pathologie_enc', 'is_smoker', 'spo2_trend', 'bpm_trend', 'spo2_volatility']

X = df[features]
y = df['target']

# On utilise scale_pos_weight car les crises sont plus rares que la stabilité
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=250, 
    max_depth=7, 
    learning_rate=0.03, 
    scale_pos_weight=4.0, # Donne plus d'importance aux détections de crises
    objective='binary:logistic'
)

model.fit(X_train, y_train)

# 5. Sauvegarde
model_dir = 'ml_engine/models/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model.save_model(os.path.join(model_dir, 'respiratory_model_predictive.json'))

print(f" Modèle mis à jour avec succès !")
print(f" Volume d'entraînement : {len(df)} mesures analysées.")