import psycopg2
import pandas as pd
import os
import logging
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime, timedelta
from ml_engine.predictor import RespiratoryAI
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ai_engine = None
latest_results = {}

try:
    ai_engine = RespiratoryAI()
    logger.info("IA SmartBreath connectée et prête (Température incluse)")
except Exception as e:
    logger.error(f"Erreur IA : {e}")

app = FastAPI(title="SmartBreath Proactive API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

# --- MODÈLES DE DONNÉES ---

class RespiratoryMeasure(BaseModel):
    patient_id: str
    flow_rate: float
    muscle_strength: float
    spo2: float
    bpm: int
    temperature: float  

class UserRegister(BaseModel):
    nom: str
    prenom: str
    email: EmailStr
    password: str
    date_naissance: str
    sexe: str = "M"
    taille_cm: int
    poids_kg: float
    pathologie: str = "Non spécifié"
    est_fumeur: bool = False

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ProfileUpdate(BaseModel):
    taille_cm: int
    poids_kg: float
    pathologie: str

# --- FONCTIONS UTILITAIRES ---

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def save_to_db(patient_id, measure, risk_score, status, recommendation):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        temp_value = float(measure.temperature)
        query = """
            INSERT INTO sensor_data 
            (patient_id, spo2, bpm, flow_rate, muscle_strength, temperature, risk_score, status, recommendation, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cur.execute(query, (
            patient_id, measure.spo2, measure.bpm, measure.flow_rate, 
            measure.muscle_strength,temp_value, measure.temperature, risk_score, 
            status, recommendation, datetime.now()
        ))
        conn.commit()
        cur.close(); conn.close()
    except Exception as e:
        logger.error(f"Erreur SQL Save : {e}")

def get_patient_context(patient_id):
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("SELECT age, taille_cm, pathologie, nom, prenom, est_fumeur, poids_kg FROM patients WHERE patient_id = %s", (patient_id,))
        res = cur.fetchone(); cur.close(); conn.close()
        if res:
            return {"age": res[0], "height": res[1], "pathologie": res[2], "nom": res[3], "prenom": res[4], "is_smoker": bool(res[5]), "weight": res[6]}
    except: pass
    return {"age": 45, "height": 170, "pathologie": "Non spécifié", "nom": "Patient", "prenom": "", "is_smoker": False, "weight": 70}

def generate_mobile_response(status, recommendation, spo2):
    status_config = {
        "CRITIQUE": {"color": "red", "vibrate": True, "emergency": True},
        "PRÉVENTION": {"color": "orange", "vibrate": False, "emergency": False},
        "SURVEILLANCE": {"color": "yellow", "vibrate": False, "emergency": False},
        "STABLE": {"color": "green", "vibrate": False, "emergency": False}
    }
    return status_config.get(status, {"color": "grey", "vibrate": False, "emergency": False})

# --- ENDPOINTS AUTH & PROFILE ---

@app.post("/register")
async def register(user: UserRegister):
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("""
        INSERT INTO patients (nom, prenom, email, password, date_naissance, sexe, taille_cm, poids_kg, pathologie, est_fumeur)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING patient_id;
    """, (user.nom, user.prenom, user.email, user.password, user.date_naissance, user.sexe, user.taille_cm, user.poids_kg, user.pathologie, user.est_fumeur))
    new_id = cur.fetchone()[0]
    conn.commit(); cur.close(); conn.close()
    return {"status": "success", "patient_id": str(new_id)}

@app.post("/login")
async def login(credentials: UserLogin):
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("SELECT patient_id, nom, password FROM patients WHERE email = %s", (credentials.email,))
    user = cur.fetchone(); cur.close(); conn.close()
    if user and user[2] == credentials.password:
        return {"status": "success", "patient_id": str(user[0]), "nom": user[1]}
    raise HTTPException(status_code=401, detail="Identifiants incorrects")

# --- ANALYSE IA ---

@app.post("/analyze")
async def analyze(measure: RespiratoryMeasure):
    global ai_engine 
    if ai_engine is None: raise HTTPException(status_code=503, detail="IA non prête")
    
    ctx = get_patient_context(measure.patient_id)
    ai_input = {**measure.dict(), **ctx, "patient_id": measure.patient_id}
    ai_res = ai_engine.predict(ai_input)
    
    risk_score = ai_res.get('risk_score', 0.5)
    status = ai_res.get('status', 'STABLE')
    recommendation = ai_res.get('recommendation', 'Analyse terminée')
    mobile_content = generate_mobile_response(status, recommendation, measure.spo2)
    
    res_payload = {
        "status": status, "risk_score": float(risk_score), "spo2": measure.spo2,
        "bpm": measure.bpm, "temperature": measure.temperature,
        "recommendation": recommendation, **mobile_content,
        "timestamp": datetime.now().isoformat()
    }
    
    latest_results[measure.patient_id] = res_payload
    save_to_db(measure.patient_id, measure, risk_score, status, recommendation)
    return res_payload

# --- STATS & DASHBOARD (DYNAMIQUE) ---

@app.get("/dashboard-summary/{patient_id}")
async def get_dashboard_summary(patient_id: str):
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("""
        SELECT ROUND(AVG(spo2)::numeric, 1), ROUND((AVG(risk_score) * 100)::numeric, 1),
        COUNT(*) FILTER (WHERE status = 'CRITIQUE'), COUNT(*) FILTER (WHERE status = 'PRÉVENTION'), COUNT(*)
        FROM sensor_data WHERE patient_id = %s AND timestamp > NOW() - INTERVAL '24 hours'
    """, (patient_id,))
    row = cur.fetchone(); cur.close(); conn.close()
    return {
        "spo2_moyen": float(row[0] or 0), "risque_moyen": float(row[1] or 0),
        "nb_alertes_critiques": int(row[2] or 0), "nb_alertes_preventives": int(row[3] or 0),
        "total_mesures": int(row[4] or 0)
    }

@app.get("/stats/{patient_id}")
async def get_stats_dynamique(patient_id: str, periode: str = "semaine"):
    intervals = {"semaine": "7 days", "mois": "30 days", "annee": "1 year"}
    interval = intervals.get(periode, "7 days")

    conn = get_db_connection(); cur = conn.cursor()
    
    # 1. Risque moyen actuel
    cur.execute(f"SELECT AVG(risk_score) FROM sensor_data WHERE patient_id = %s AND timestamp > NOW() - INTERVAL '{interval}'", (patient_id,))
    actuel = cur.fetchone()[0] or 0

    # 2. Amélioration (vs période précédente)
    cur.execute(f"SELECT AVG(risk_score) FROM sensor_data WHERE patient_id = %s AND timestamp BETWEEN NOW() - INTERVAL '2 {interval}' AND NOW() - INTERVAL '{interval}'", (patient_id,))
    precedent = cur.fetchone()[0] or actuel
    diff = round((precedent - actuel) * 100, 1) 

    # 3. Données du graphique (Risk evolution)
    cur.execute(f"""
        SELECT TO_CHAR(timestamp, 'DD/MM'), AVG(risk_score) * 100 
        FROM sensor_data WHERE patient_id = %s AND timestamp > NOW() - INTERVAL '{interval}'
        GROUP BY 1 ORDER BY MIN(timestamp) ASC
    """, (patient_id,))
    graph_rows = cur.fetchall()

    # 4. Succès/Badges Dynamiques
    cur.execute("SELECT COUNT(*), MAX(spo2), MIN(risk_score) FROM sensor_data WHERE patient_id = %s", (patient_id,))
    totals = cur.fetchone()
    badges = {
        "pionnier": totals[0] >= 1,
        "expert": totals[0] >= 50,
        "poumon_acier": (totals[1] or 0) >= 98,
        "zen": (totals[2] or 1) < 0.2
    }

    cur.close(); conn.close()
    return {
        "risque_moyen": round(float(actuel) * 100, 1),
        "amelioration_pourcent": diff,
        "jours_consecutifs": totals[0],
        "alertes_preventives": 0, # À calculer selon status
        "crises": 0,
        "badges": badges,
        "graph_data": {
            "labels": [r[0] for r in graph_rows] if graph_rows else ["N/A"],
            "values": [float(r[1]) for r in graph_rows] if graph_rows else [0]
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "ia": "ready" if ai_engine else "off"}