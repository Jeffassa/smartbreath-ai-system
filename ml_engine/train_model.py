import xgboost as xgb
import pandas as pd
import numpy as np
import os

def generate_medical_data(n_samples=5000):
    """
    Génère des données synthétiques cohérentes pour l'entraînement.
    Note : L'ordre des colonnes ici doit être respecté par le Predictor.
    """
    np.random.seed(42)

    data = {
        'spo2': np.random.uniform(85, 100, n_samples),
        'bpm': np.random.uniform(60, 120, n_samples),
        'muscle_strength': np.random.uniform(10, 100, n_samples),
        'flow_rate': np.random.uniform(1.0, 5.0, n_samples),
        'age': np.random.randint(18, 90, n_samples),
        'height': np.random.randint(150, 200, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Logique de ciblage (Target) : 
    # 1 = Risque élevé (Crise), 0 = Normal
    condition = (
        (df['spo2'] < 92) | 
        (df['flow_rate'] < 2.0) | 
        ((df['spo2'] < 94) & (df['bpm'] > 105))
    )
    df['target'] = condition.astype(int)
    
    return df

print("--- Phase 1 : Génération des données médicales ---")
df = generate_medical_data()

# Définition explicite de l'ordre des caractéristiques
feature_names = ['spo2', 'bpm', 'muscle_strength', 'flow_rate', 'age', 'height']
X = df[feature_names]
y = df['target']

print(f"Dataset prêt : {X.shape[0]} échantillons, {X.shape[1]} variables.")

print("--- Phase 2 : Entraînement XGBoost ---")
# Conversion en DMatrix avec noms de colonnes pour plus de robustesse
dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)

param = {
    'max_depth': 4,             
    'eta': 0.1,                 
    'objective': 'binary:logistic', 
    'eval_metric': 'auc',
    'random_state': 42
}

bst = xgb.train(param, dtrain, num_boost_round=100)

print("--- Phase 3 : Sauvegarde du modèle ---")
os.makedirs('ml_engine/models', exist_ok=True)
model_path = 'ml_engine/models/respiratory_model.json'
bst.save_model(model_path)

print(f"Succès ! Modèle cohérent sauvegardé dans : {model_path}")
print(f"Ordre des variables enregistré : {feature_names}")