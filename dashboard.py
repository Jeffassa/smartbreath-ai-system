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
        # Test de connexion
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f" Erreur de configuration DB : {e}")
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
                spo2, 
                bpm, 
                temperature,
                flow_rate,
                muscle_strength, 
                risk_score,
                status,
                recommendation,
                timestamp AT TIME ZONE 'UTC' as timestamp
            FROM sensor_data 
            WHERE patient_id::text = :p_id 
            ORDER BY timestamp DESC 
            LIMIT 60
        """)
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"p_id": str(p_id)})
        
        if df.empty:
            return pd.DataFrame()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        st.error(f"Erreur SQL : {e}")
        return pd.DataFrame()

def check_connection_status(last_timestamp):
    if pd.isna(last_timestamp):
        return "🔴 AUCUNE DONNÉE", "Pas de données reçues"
    
    time_diff = (datetime.now() - last_timestamp).total_seconds()
    if time_diff > 30: 
        return "🔴 DÉCONNECTÉ", f"Dernière mesure il y a {int(time_diff)}s"
    return "🟢 EN LIGNE", "Données reçues en temps réel"

# --- SIDEBAR & SÉLECTION ---
st.sidebar.title("Dashboard Medecin - SmartBreath AI")
engine = get_engine()

if not engine:
    st.error("Impossible de se connecter à la base de données")
    st.stop()

try:
    query_pats = text("SELECT patient_id::text as patient_id, nom, prenom, email FROM patients ORDER BY nom ASC")
    with engine.connect() as conn:
        df_pats = pd.read_sql(query_pats, conn)
    
    if df_pats.empty:
        st.sidebar.warning("Aucun patient trouvé")
        st.stop()
    
    patient_dict = dict(zip(
        df_pats['nom'] + " " + df_pats['prenom'] + " (" + df_pats['email'] + ")", 
        df_pats['patient_id']
    ))
    
    selected_name = st.sidebar.selectbox("Choisir un patient :", list(patient_dict.keys()))
    selected_id = patient_dict[selected_name]
    
except Exception as e:
    st.sidebar.error(f"Erreur chargement patients : {e}")
    selected_id = None

# --- AFFICHAGE PRINCIPAL ---
if selected_id:
    patient_info = get_patient_details(selected_id)
    user_data = get_live_data(selected_id)

    if user_data.empty:
        st.warning(f"En attente de données pour {selected_name.split('(')[0]}...")
    elif patient_info is None:
        st.error(f"Patient introuvable")
    else:
        last = user_data.iloc[-1]
        status_label, conseil = check_connection_status(last['timestamp'])
        
        # Titre et Alerte Critique
        st.title(f"Monitoring : {patient_info['nom']} {patient_info['prenom']}")
        
        is_critique = str(last.get('status', '')).upper() == "CRITIQUE" or last['spo2'] < 90
        if is_critique:
            st.error(f"ALERTE CRITIQUE : {last.get('recommendation')}")

        # --- MÉTRIQUES CLÉS ---
        st.write("---")
        m1, m2, m3, m4, m5 = st.columns(5)
        
        # Oxygène
        m1.metric("Oxygène (SpO2)", f"{last['spo2']}%", 
                  delta=f"{last['spo2']-95:.1f}%" if last['spo2'] < 95 else None, delta_color="inverse")
        
        # Pouls
        m2.metric("Pouls (BPM)", f"{int(last['bpm'])} bpm")
        
        # TEMPÉRATURE
        temp_val = last.get('temperature', 36.6)
        temp_delta = round(temp_val - 36.6, 1)
        m3.metric("Température", f"{temp_val}°C", 
                  delta=f"{temp_delta}°C" if abs(temp_delta) > 0.2 else None, 
                  delta_color="inverse" if temp_val > 37.5 else "normal")
        
        # Autres
        m4.metric("Force Musc.", f"{last.get('muscle_strength', 'N/A')}")
        m5.metric("Débit d'air", f"{last.get('flow_rate', 'N/A')} L/m")

        # --- GRAPHIQUES ---
        st.subheader("Courbes Physiologiques")
        
        if len(user_data) > 1:
            plt.style.use('dark_background')
            fig, ax1 = plt.subplots(figsize=(12, 5))
            fig.patch.set_facecolor('#0E1117')
            ax1.set_facecolor('#1e2129')

            # SpO2 sur l'axe principal
            ax1.plot(user_data['timestamp'], user_data['spo2'], 
                    color='#00d4ff', label='SpO2 (%)', linewidth=2, marker='o', markersize=4)
            ax1.set_ylabel('% SpO2', color='#00d4ff')
            ax1.set_ylim(min(75, user_data['spo2'].min()-5), 102)

            # BPM sur le second axe
            ax2 = ax1.twinx()
            ax2.plot(user_data['timestamp'], user_data['bpm'], 
                    color='#ff4b4b', label='BPM', linewidth=1.5, linestyle='--')
            ax2.set_ylabel('BPM', color='#ff4b4b')
            
            # TEMPÉRATURE sur un troisième axe (tendance)
            ax2.plot(user_data['timestamp'], user_data['temperature'] * 2, 
                    color='#ffaa00', label='Temp (Tendance)', linewidth=1, alpha=0.6)

            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.xticks(rotation=45)
            
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left')
            
            st.pyplot(fig)
            plt.close(fig)

        # --- PROFIL & ANALYSE IA ---
        col_p, col_ia = st.columns(2)
        with col_p:
            st.subheader("Profil Patient")
            st.write(f"**Pathologie :** :orange[{patient_info.get('pathologie')}]")
            st.write(f"**Fumeur :** {'Oui' if patient_info.get('est_fumeur') else 'Non'}")
            st.write(f"**Dernière activité :** {last['timestamp'].strftime('%H:%M:%S')}")
            
        with col_ia:
            st.subheader("Analyse SmartBreath")
            risk_pct = round(float(last.get('risk_score', 0)) * 100)
            color_risk = "green" if risk_pct < 30 else "orange" if risk_pct < 70 else "red"
            st.markdown(f"Probabilité de crise : <span style='color:{color_risk}; font-size:24px; font-weight:bold;'>{risk_pct}%</span>", unsafe_allow_html=True)
            st.info(f"**Recommandation :** {last.get('recommendation')}")

        # --- HISTORIQUE DÉTAILLÉ ---
        with st.expander("Historique complet (60 dernières mesures)"):
            cols_to_show = ['timestamp', 'spo2', 'bpm', 'temperature', 'status', 'recommendation']
            # CORRECTION 2026 : width='stretch' remplace use_container_width=True
            st.dataframe(user_data.sort_values('timestamp', ascending=False)[cols_to_show], width='stretch')

else:
    st.info("Sélectionnez un patient dans la barre latérale pour commencer le monitoring.")