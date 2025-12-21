import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import base64
from streamlit_autorefresh import st_autorefresh
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
from sqlalchemy import create_engine
import urllib.parse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Chargement des variables d'environnement
load_dotenv()

# Configuration de la page
st.set_page_config(page_title="IA Respiratoire - Monitoring", layout="wide")

# Actualisation automatique toutes les 2 secondes
st_autorefresh(interval=2000, key="datarefresh")

def play_alarm():
    """Diffuse un son d'alerte via HTML"""
    sound_html = """
        <audio autoplay>
            <source src="https://www.soundjay.com/mechanical/smoke-detector-1.mp3" type="audio/mpeg">
        </audio>
    """
    st.markdown(sound_html, unsafe_allow_html=True)

# Styles CSS personnalisés (Clignotement URGENCE)
st.markdown("""
    <style>
    @keyframes blinker { 50% { opacity: 0; } }
    .blink-emergency {
        background-color: #FF0000; color: white; padding: 15px;
        border-radius: 10px; text-align: center; font-weight: bold;
        font-size: 20px; animation: blinker 1s linear infinite; margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration de la connexion Base de Données
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

@st.cache_resource
def get_engine():
    """Crée l'engine SQLAlchemy avec gestion des caractères spéciaux dans le MDP"""
    user = DB_CONFIG['user']
    # Encode le mot de passe pour gérer les caractères comme @ ou !
    password = urllib.parse.quote_plus(str(DB_CONFIG['password'])) 
    host = DB_CONFIG['host']
    db = DB_CONFIG['database']
    conn_url = f"postgresql+psycopg2://{user}:{password}@{host}:5432/{db}"
    return create_engine(conn_url)

def get_patients_list():
    """Récupère la liste des patients pour la barre latérale"""
    try:
        engine = get_engine()
        query = "SELECT patient_id, nom, prenom FROM patients ORDER BY nom ASC"
        df = pd.read_sql(query, engine)
        df['display_name'] = df['nom'].str.upper() + " " + df['prenom']
        return df
    except Exception as e:
        st.error(f"Erreur Base de Données : {e}")
        return pd.DataFrame()

def generate_pdf(df_patient, patient_name):
    """Génère un rapport PDF avec graphique et statistiques"""
    buffer = BytesIO()
    
    # 1. Création de la figure Matplotlib
    fig, ax = plt.subplots(figsize=(6, 3)) 
    recent_data = df_patient.tail(30)
    
    ax.plot(recent_data['timestamp'], recent_data['spo2'], color='blue', label='SpO2 %')
    ax.plot(recent_data['timestamp'], recent_data['bpm'], color='red', label='BPM')
    
    ax.set_title(f"Signes vitaux - {patient_name}")
    ax.set_xticks([]) 
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 2. Sauvegarde de l'image en mémoire
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight')
    plt.close(fig) 
    img_buffer.seek(0)
    
    # 3. Construction du PDF
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, 750, f"RAPPORT MEDICAL : {patient_name}")
    
    p.setFont("Helvetica", 10)
    p.drawString(100, 735, f"Genere le : {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}")
    
    p.drawImage(ImageReader(img_buffer), 50, 450, width=500, height=250)
    
    p.setFont("Helvetica-Bold", 12)
    p.drawString(100, 420, "Resume des dernieres constantes :")
    p.setFont("Helvetica", 11)
    p.drawString(100, 400, f"- SpO2 Moyenne : {round(df_patient['spo2'].mean(),1)}%")
    p.drawString(100, 380, f"- Rythme Cardiaque Max : {int(df_patient['bpm'].max())} BPM")
    
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

# --- BARRE LATÉRALE ---
st.sidebar.header("Gestion des Dossiers")
df_pats = get_patients_list()
selected_id = None

if not df_pats.empty:
    patient_dict = dict(zip(df_pats['display_name'], df_pats['patient_id']))
    selected_name = st.sidebar.selectbox("Choisir le patient :", list(patient_dict.keys()))
    selected_id = str(patient_dict[selected_name])
    
    if os.path.exists("live_data.csv"):
        full_live = pd.read_csv("live_data.csv")
        p_data = full_live[full_live['patient_id'].astype(str) == selected_id]
        if not p_data.empty:
            pdf_file = generate_pdf(p_data, selected_name)
            st.sidebar.download_button(
                label="Telecharger Rapport PDF",
                data=pdf_file,
                file_name=f"rapport_{selected_name.replace(' ','_')}.pdf",
                mime="application/pdf"
            )
else:
    st.sidebar.error("Impossible de charger la liste des patients.")

# --- ZONE PRINCIPALE ---
st.title("SmartBreath AI - Monitoring")

if os.path.exists("live_data.csv") and selected_id:
    df_live = pd.read_csv("live_data.csv") 
    user_data = df_live[df_live['patient_id'].astype(str) == selected_id].copy()

    if not user_data.empty:
        last = user_data.iloc[-1]
        status_ia = str(last.get('status', 'NORMAL')).upper()
        
        # Détection Urgence
        if last['bpm'] > 120 or "CRITIQUE" in status_ia:
            st.markdown(f'<div class="blink-emergency">URGENCE CRITIQUE : {selected_name} - Intervention Immediate</div>', unsafe_allow_html=True)
            play_alarm() 

        # Métriques principales
        c1, c2, c3 = st.columns(3)
        c1.metric("SpO2", f"{last['spo2']}%")
        c2.metric("Fréquence Cardiaque", f"{int(last['bpm'])} BPM")
        c3.metric("Niveau de Risque IA", f"{round(last.get('risk_score', 0)*100)}%")

        st.divider()
        st.subheader("Analyse Prédictive de l'IA")
        
        # Affichage selon le statut IA
        if "CRITIQUE" in status_ia:
            st.error(f"**STATUT : {status_ia}** \n\n {last['recommendation']}")
        elif "MODERE" in status_ia:
            st.warning(f"**STATUT : {status_ia}** \n\n {last['recommendation']}")
        else:
            st.success(f"**STATUT : {status_ia}** \n\n {last['recommendation']}")

        # Time To Crisis (TTC)
        ttc = last.get('time_to_crisis')
        if pd.notna(ttc) and ttc > 0:
            st.info(f"**Alerte Anticipee :** Risque de crise d'ici environ **{ttc} minutes**.")

        # Graphiques dynamiques
        st.subheader("Evolution des Signes Vitaux (Temps Reel)")
        graph_data = user_data.tail(50).set_index('timestamp')[['spo2', 'bpm']]
        st.line_chart(graph_data)
        
    else:
        st.info(f"En attente de réception des données capteurs pour **{selected_name}**...")
else:
    st.warning("Veuillez sélectionner un patient dans la barre latérale pour lancer le monitoring.")