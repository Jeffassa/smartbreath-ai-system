import time
import psycopg2
import pandas as pd
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, date
from ml_engine.predictor import RespiratoryAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("system.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

try:
    ai_engine = RespiratoryAI()
    logger.info("Moteur IA charge avec succes")
except Exception as e:
    logger.error(f"Erreur chargement IA : {e}")

app = FastAPI(title="Smart Respiratory AI System")

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

def get_patient_profile(patient_id):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        query = "SELECT date_naissance, taille_cm FROM patients WHERE patient_id = %s"
        cur.execute(query, (patient_id,))
        result = cur.fetchone()
        cur.close()
        conn.close()
        if result:
            dob, height = result
            today = date.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            return age, height
    except Exception as e:
        logger.error(f"Erreur SQL profil : {e}")
    return None, None

def analyze_vital_trends(patient_id, current_spo2, current_bpm):
    analysis = {"ttc": None, "warning": None}
    if current_bpm > 120:
        analysis["warning"] = f"Alerte : Rythme cardiaque critique ({current_bpm} BPM)"
    try:
        if os.path.exists("live_data.csv"):
            df_hist = pd.read_csv("live_data.csv").tail(10)
            df_p = df_hist[df_hist['patient_id'].astype(str) == str(patient_id)]
            if len(df_p) >= 1:
                pente = current_spo2 - df_p['spo2'].iloc[-1]
                if pente < -0.01 and current_spo2 > 90:
                    minutes = (90 - current_spo2) / pente
                    analysis["ttc"] = round(max(0, minutes), 1)
    except Exception as e:
        logger.warning(f"Erreur tendance : {e}")
    return analysis

class RespiratoryMeasure(BaseModel):
    patient_id: str
    flow_rate: float
    muscle_strength: float
    spo2: float
    bpm: int

@app.post("/analyze")
async def analyze_breath(measure: RespiratoryMeasure):
    try:
        age, height = get_patient_profile(measure.patient_id)
        if age is None:
            return {"status": "error", "condition": "ERREUR IA", "recommendation": "Patient inconnu"}

        ai_input = {"spo2": measure.spo2, "bpm": measure.bpm, "muscle_strength": measure.muscle_strength, "flow_rate": measure.flow_rate, "age": age, "height": height}
        ai_result = ai_engine.predict(ai_input)
        
        vitals = analyze_vital_trends(measure.patient_id, measure.spo2, measure.bpm)
        final_rec = ai_result["recommendation"]
        if vitals["warning"]: final_rec = f"{vitals['warning']} | {final_rec}"

        # Sauvegarde SQL
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            cur.execute("INSERT INTO sensor_data (patient_id, spo2, bpm, airflow_pressure) VALUES (%s, %s, %s, %s)", 
                        (measure.patient_id, measure.spo2, measure.bpm, measure.flow_rate))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e: logger.error(f"SQL Save Error : {e}")

        # Log CSV
        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "patient_id": str(measure.patient_id), "spo2": measure.spo2, "bpm": measure.bpm,
            "risk_score": round(ai_result.get("risk_score", 0), 4), "status": ai_result.get("status", "NORMAL"),
            "recommendation": final_rec, "time_to_crisis": vitals["ttc"]
        }
        pd.DataFrame([log_data]).to_csv("live_data.csv", mode='a', header=not os.path.exists("live_data.csv"), index=False)
        
        return {"status": "success", "risk_score": ai_result.get("risk_score", 0), "condition": ai_result.get("status", "NORMAL"), "ttc": vitals["ttc"]}
    except Exception as e:
        logger.error(f"Erreur critique : {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)