import streamlit as st
import pandas as pd
import os
import urllib.parse
from datetime import datetime
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import base64

# Configuration et Chargement
load_dotenv()
st.set_page_config(page_title="SmartBreath AI - Expert Dashboard", layout="wide")
st_autorefresh(interval=2000, key="datarefresh") 

@st.cache_resource
def get_engine():
    try:
        user = os.getenv("DB_USER")
        password = urllib.parse.quote_plus(str(os.getenv("DB_PASSWORD")))
        host = os.getenv("DB_HOST")
        db = os.getenv("DB_NAME")
        conn_url = f"postgresql+psycopg2://{user}:{password}@{host}:5432/{db}"
        engine = create_engine(conn_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"Erreur de configuration DB : {e}")
        return None

def get_patient_details(p_id):
    engine = get_engine()
    if engine:
        try:
            query = text("SELECT * FROM patients WHERE patient_id::text = :p_id")
            with engine.connect() as conn:
                df = pd.read_sql(query, conn, params={"p_id": str(p_id)})
                if not df.empty:
                    return df.iloc[0]
        except Exception as e:
            st.error(f"Erreur détails patient : {e}")
    return None

def get_live_data(p_id):
    engine = get_engine()
    if not engine: 
        return pd.DataFrame() 
    
    try:
        query = text("""
            SELECT 
                patient_id::text as patient_id,
                spo2, bpm, temperature, flow_rate, muscle_strength, 
                risk_score, status, recommendation,
                actual_outcome, feedback_notes,
                timestamp AT TIME ZONE 'UTC' as timestamp
            FROM sensor_data 
            WHERE patient_id::text = :p_id 
            ORDER BY timestamp DESC 
            LIMIT 60
        """)
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"p_id": str(p_id)})
        
        if df is None or df.empty:
            return pd.DataFrame()
  
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df

    except Exception as e:
        st.error(f"Erreur SQL dans get_live_data : {e}")
        return pd.DataFrame()

def check_connection_status(last_timestamp):
    if pd.isna(last_timestamp):
        return "🔴 AUCUNE DONNÉE", "Pas de données reçues"
    time_diff = (datetime.now() - last_timestamp).total_seconds()
    if time_diff > 30: 
        return "🔴 DÉCONNECTÉ", f"Dernière mesure il y a {int(time_diff)}s"
    return "🟢 EN LIGNE", "Données reçues en temps réel"

# --- SIDEBAR ---
st.sidebar.title("SmartBreath AI - Médical")
engine = get_engine()

if not engine:
    st.error("Impossible de se connecter à la base de données")
    st.stop()

try:
    query_pats = text("SELECT patient_id::text as patient_id, nom, prenom, email FROM patients ORDER BY nom ASC")
    with engine.connect() as conn:
        df_pats = pd.read_sql(query_pats, conn)
    
    patient_dict = dict(zip(
        df_pats['nom'] + " " + df_pats['prenom'] + " (" + df_pats['email'] + ")", 
        df_pats['patient_id']
    ))
    selected_name = st.sidebar.selectbox("Choisir un patient :", list(patient_dict.keys()))
    selected_id = patient_dict[selected_name]
except Exception as e:
    st.sidebar.error(f"Erreur : {e}")
    selected_id = None

if selected_id:
    patient_info = get_patient_details(selected_id)
    user_data = get_live_data(selected_id)

    if user_data.empty:
        st.warning(f"En attente de données pour {selected_name}...")
    elif patient_info is not None:
        last = user_data.iloc[-1]
        status_label, connection_msg = check_connection_status(last['timestamp'])
        
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            st.title(f"Monitoring : {patient_info['nom']} {patient_info['prenom']}")
        with col_t2:
            st.subheader(status_label)
            st.caption(connection_msg)

        # Alerte Critique (IA + Seuil de sécurité)
        is_critique = str(last.get('status', '')).upper() == "CRITIQUE" or last['spo2'] < 90
        
        if is_critique:
            st.error(f" ALERTE CRITIQUE : {last.get('recommendation')}")
            
           
            sound_html = """
                <div style="display:none;">
                    <audio id="alarm-audio" autoplay loop>
                        <source src="https://actions.google.com/sounds/v1/alarms/alarm_clock_short.ogg" type="audio/ogg">
                        <source src="https://www.soundjay.com/buttons/beep-01a.mp3" type="audio/mpeg">
                    </audio>
                    <script>
                        var audio = document.getElementById("alarm-audio");
                        audio.volume = 1.0;
                        // Forcer la lecture si le navigateur bloque
                        document.addEventListener('click', function() {
                            audio.play();
                        }, { once: true });
                    </script>
                </div>
            """
            st.components.v1.html(sound_html, height=0)
        st.write("---")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Oxygène (SpO2)", f"{last['spo2']}%", delta=f"{last['spo2']-95:.1f}%" if last['spo2'] < 95 else None, delta_color="inverse")
        m2.metric("Pouls (BPM)", f"{int(last['bpm'])} bpm")
        
        temp_val = last.get('temperature', 36.6)
        m3.metric("Température", f"{temp_val}°C", delta=f"{round(temp_val-36.6,1)}°C" if abs(temp_val-36.6) > 0.2 else None, delta_color="inverse" if temp_val > 37.5 else "normal")
        
        m4.metric("Force Musc.", f"{last.get('muscle_strength', 'N/A')}")
        m5.metric("Débit d'air", f"{last.get('flow_rate', 'N/A')} L/m")

        st.subheader("Courbes Physiologiques (Temps Réel)")
        if len(user_data) > 1:
            plt.style.use('dark_background')
            fig, ax1 = plt.subplots(figsize=(12, 4))
            fig.patch.set_facecolor('#0E1117')
            ax1.set_facecolor('#1e2129')

            ax1.plot(user_data['timestamp'], user_data['spo2'], color='#00d4ff', label='SpO2 (%)', linewidth=2)
            ax1.set_ylabel('% SpO2', color='#00d4ff')
            ax1.set_ylim(min(80, user_data['spo2'].min()-5), 102)

            ax2 = ax1.twinx()
            ax2.plot(user_data['timestamp'], user_data['bpm'], color='#ff4b4b', label='BPM', linewidth=1.5, linestyle='--')
            ax2.set_ylabel('BPM', color='#ff4b4b')

            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            st.pyplot(fig)
            plt.close(fig)
            
        st.write("---")
        col_ia, col_p = st.columns(2)
        
        with col_ia:
            st.subheader("Intelligence Artificielle")
            risk_pct = round(float(last.get('risk_score', 0)) * 100)
            
            feedback = last.get('actual_outcome')
            if feedback == 1: 
                color_risk = "red"
                fb_msg = " LE PATIENT CONFIRME LA DÉTRESSE"
            elif feedback == 0: 
                color_risk = "green"
                fb_msg = " LE PATIENT DÉCLARE ALLER BIEN (Fausse alerte)"
            else:
                color_risk = "green" if risk_pct < 30 else "orange" if risk_pct < 70 else "red"
                fb_msg = " En attente du feedback patient..."

            st.markdown(f"Probabilité de crise : <span style='color:{color_risk}; font-size:32px; font-weight:bold;'>{risk_pct}%</span>", unsafe_allow_html=True)
            st.info(f"**Ressenti Patient :** {fb_msg}")
            if last.get('feedback_notes'):
                st.write(f" *Note : {last['feedback_notes']}*")

        with col_p:
            st.subheader("Détails Cliniques")
            st.write(f"**Pathologie :** {patient_info.get('pathologie')}")
            st.write(f"**Fumeur :** {'Oui' if patient_info.get('est_fumeur') else 'Non'}")
            st.write(f"**Dernière mesure :** {last['timestamp'].strftime('%H:%M:%S')}")

        # --- TABLEAU D'HISTORIQUE ---
        with st.expander("Consulter l'historique détaillé (60 dernières mesures)"):
            df_hist = user_data.sort_values('timestamp', ascending=False).copy()
            df_hist['Vérité Terrain'] = df_hist['actual_outcome'].map({1: "Crise Confirmée", 0: "Stable (Fausse Alerte)"}).fillna("Non renseigné")
            
            cols = ['timestamp', 'spo2', 'bpm', 'temperature', 'status', 'Vérité Terrain', 'feedback_notes']
            st.dataframe(df_hist[cols], width='stretch')

else:
    st.info("Sélectionnez un patient pour démarrer le monitoring.")