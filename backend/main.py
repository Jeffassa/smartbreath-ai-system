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

class RespiratoryMeasure(BaseModel):
    patient_id: str
    flow_rate: float
    muscle_strength: float
    spo2: float
    bpm: int
    temperature: float  

class FeedbackData(BaseModel):
    data_id: int
    actual_outcome: int 
    comment: Optional[str] = None

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
    taille_cm: Optional[int] = None
    poids_kg: Optional[float] = None
    pathologie: Optional[str] = None
    photo_base64: Optional[str] = None


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def save_to_db(patient_id, measure, risk_score, status, recommendation):
    """Insère la mesure et retourne l'ID généré pour le feedback futur"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        temp_value = float(measure.temperature)
        
        query = """
            INSERT INTO sensor_data 
            (patient_id, timestamp, spo2, bpm, flow_rate, muscle_strength, risk_score, status, recommendation, temperature)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING data_id
        """
        cur.execute(query, (
            patient_id, datetime.now(), measure.spo2, measure.bpm, measure.flow_rate, 
            measure.muscle_strength, risk_score, 
            status, recommendation, temp_value, 
        ))
        generated_id = cur.fetchone()[0]
        conn.commit()
        cur.close(); conn.close()
        return generated_id
    except Exception as e:
        logger.error(f"Erreur SQL Save : {e}")
        return None

def get_patient_context(patient_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT age, taille_cm, pathologie, nom, prenom, est_fumeur, poids_kg, email, photo_base64 
            FROM patients WHERE patient_id = %s
        """, (patient_id,))
        res = cur.fetchone()
        cur.close()
        conn.close()
        if res:
            return {
                "age": res[0], "height": res[1], "pathologie": res[2], 
                "nom": res[3], "prenom": res[4], "is_smoker": bool(res[5]), 
                "weight": res[6], "email": res[7],
                "photo_url": res[8] 
            }
    except Exception as e:
        logger.error(f"Erreur contexte patient : {e}")
    return {"nom": "Patient", "photo_url": None}

def generate_mobile_response(status, recommendation, spo2):
    status_config = {
        "CRITIQUE": {"color": "red", "vibrate": True, "emergency": True},
        "PRÉVENTION": {"color": "orange", "vibrate": False, "emergency": False},
        "SURVEILLANCE": {"color": "yellow", "vibrate": False, "emergency": False},
        "STABLE": {"color": "green", "vibrate": False, "emergency": False}
    }
    return status_config.get(status, {"color": "grey", "vibrate": False, "emergency": False})


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
    
    data_id = save_to_db(measure.patient_id, measure, risk_score, status, recommendation)
    
    res_payload = {
        "data_id": data_id,
        "status": status, 
        "risk_score": float(risk_score), 
        "spo2": measure.spo2,
        "bpm": measure.bpm, 
        "temperature": measure.temperature,
        "recommendation": recommendation, 
        **mobile_content,
        "timestamp": datetime.now().isoformat()
    }
    return res_payload

@app.post("/feedback")
async def submit_feedback(fb: FeedbackData):
    """Permet au patient de confirmer ou d'infirmer l'analyse de l'IA (Apprentissage supervisé)"""
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("""
            UPDATE sensor_data 
            SET actual_outcome = %s, feedback_notes = %s 
            WHERE data_id = %s
        """, (fb.actual_outcome, fb.comment, fb.data_id))
        conn.commit(); cur.close(); conn.close()
        logger.info(f"Feedback reçu pour la mesure {fb.data_id} : Outcome={fb.actual_outcome}")
        return {"status": "success", "message": "Merci, SmartBreath apprend de votre expérience."}
    except Exception as e:
        logger.error(f"Erreur feedback : {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de l'enregistrement du feedback")

@app.get("/profile/{patient_id}")
async def get_profile(patient_id: str):
    ctx = get_patient_context(patient_id)
    return {"status": "success", "data": ctx}

@app.put("/profile/{patient_id}")
async def update_profile(patient_id: str, profile: ProfileUpdate):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        updates = []
        params = []

        if profile.taille_cm is not None:
            updates.append("taille_cm = %s"); params.append(profile.taille_cm)
        if profile.poids_kg is not None:
            updates.append("poids_kg = %s"); params.append(profile.poids_kg)
        if profile.pathologie is not None:
            updates.append("pathologie = %s"); params.append(profile.pathologie)
        
        if profile.photo_base64 is not None:
            updates.append("photo_base64 = %s")
            params.append(profile.photo_base64)

        if not updates:
            return {"status": "no update needed"}

        params.append(patient_id)
        query = f"UPDATE patients SET {', '.join(updates)} WHERE patient_id = %s"
        
        cur.execute(query, tuple(params))
        conn.commit()
        cur.close()
        conn.close()
        
        logger.info(f"Profil et photo mis à jour pour le patient {patient_id}")
        return {"status": "success", "message": "Profil et Photo synchronisés"}
    except Exception as e:
        logger.error(f"Erreur update profile : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{patient_id}")
async def get_status(patient_id: str):
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("""
        SELECT status, recommendation, spo2, bpm, risk_score, temperature, timestamp, data_id 
        FROM sensor_data 
        WHERE patient_id = %s 
        ORDER BY timestamp DESC LIMIT 1
    """, (patient_id,))
    res = cur.fetchone(); cur.close(); conn.close()
    
    if res:
        status_name = res[0]
        mobile_config = generate_mobile_response(status_name, res[1], res[2])
        return {
            "data_id": res[7],
            "status": status_name,
            "recommendation": res[1],
            "spo2": float(res[2]),
            "bpm": int(res[3]),
            "risk_score": float(res[4]),
            "temperature": float(res[5]),
            "color": mobile_config["color"],
            "emergency": mobile_config["emergency"],
            "last_update": res[6].isoformat()
        }
    return {"status": "STABLE", "recommendation": "Aucune donnée", "spo2": 0, "bpm": 0, "risk_score": 0, "temperature": 0}

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
    days = 7 if periode == "semaine" else 30
    if periode == "annee": days = 365
    conn = get_db_connection(); cur = conn.cursor()
    
    cur.execute("SELECT AVG(risk_score) FROM sensor_data WHERE patient_id = %s AND timestamp > NOW() - make_interval(days => %s)", (patient_id, days))
    actuel = cur.fetchone()[0] or 0

    cur.execute("""
        SELECT TO_CHAR(timestamp, 'DD/MM'), AVG(risk_score) * 100, AVG(temperature)
        FROM sensor_data WHERE patient_id = %s AND timestamp > NOW() - make_interval(days => %s)
        GROUP BY 1 ORDER BY MIN(timestamp) ASC
    """, (patient_id, days))
    graph_rows = cur.fetchall()

    cur.execute("SELECT COUNT(*), MAX(spo2), MIN(risk_score) FROM sensor_data WHERE patient_id = %s", (patient_id,))
    totals = cur.fetchone()

    cur.close(); conn.close()
    return {
        "risque_moyen": round(float(actuel) * 100, 1),
        "jours_consecutifs": totals[0],
        "graph_data": {
            "labels": [r[0] for r in graph_rows] if graph_rows else ["N/A"],
            "risk_values": [float(r[1]) for r in graph_rows] if graph_rows else [0],
            "temp_values": [float(r[2]) for r in graph_rows] if graph_rows else [0]
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "ia": "ready" if ai_engine else "off"}