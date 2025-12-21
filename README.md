🩺 SmartBreath Monitoring AI
📌 Présentation du Projet
Ce projet est une solution complète de e-santé dédiée à la surveillance intelligente des patients atteints de pathologies respiratoires (Asthme, BPCO). Le système combine l'acquisition de données physiologiques simulées, une analyse prédictive par Intelligence Artificielle et une interface de monitoring pour le personnel médical.

🌟 Fonctionnalités Clés
Analyse Prédictive : Utilisation d'un modèle XGBoost pour évaluer le risque de crise respiratoire en temps réel.

Anticipation (TTC) : Calcul du Time To Crisis pour prévenir les incidents avant qu'ils ne surviennent.

Dashboard Dynamique : Visualisation en temps réel des constantes (SpO2, BPM) via une interface Streamlit.

Génération de Rapports : Exportation automatique de bilans médicaux au format PDF via ReportLab.

Alertes Intelligentes : Système d'alerte sonore et visuelle (bandeau clignotant) en cas de détection de seuils critiques.

🏗️ Architecture Technique
Le projet repose sur une architecture découplée pour garantir performance et sécurité :

Backend (FastAPI) : Gère l'ingestion des données, la communication avec la base de données PostgreSQL et l'exécution du moteur IA.

Base de Données (PostgreSQL) : Stockage sécurisé des profils patients (identités, âges, antécédents médicaux).

Moteur IA (XGBoost) : Modèle entraîné sur 6 variables clés (SpO2, BPM, débit expiratoire, force musculaire, âge, taille).

Frontend (Streamlit) : Interface utilisateur pour le monitoring interactif et la gestion des dossiers.

🚀 Installation et Démarrage
Prérequis
Python 3.11+

PostgreSQL installé et configuré

Un environnement virtuel (venv) activé

1. Installation des dépendances
Bash

pip install -r requirements.txt
2. Entraînement du modèle
Lancez le script d'entraînement pour générer le modèle JSON basé sur les 6 caractéristiques médicales :

Bash

python ml_engine/train_model.py
3. Lancement du système (3 Terminaux)
Pour faire fonctionner la démo complète, ouvrez trois terminaux :

Terminal 1 (Backend) : python app.py

Terminal 2 (Dashboard) : streamlit run dashboard.py

Terminal 3 (Simulateur) : python mock_sensor.py

📊 Algorithme et Logique IA
Le modèle intègre une logique métier pour une IA explicable :

Score > 0.8 : Statut CRITIQUE -> Déclenchement immédiat des alarmes.

Analyse Contextuelle : L'IA ajuste son diagnostic en fonction de l'âge récupéré en base SQL (différenciation entre une crise d'asthme juvénile et une complication de BPCO chez le senior).

Sécurité : En cas de données incohérentes, le système renvoie un statut "ERREUR IA" pour garantir la sécurité du patient.

📁 Structure du Projet
Plaintext

app_backend/
├── app.py                # Serveur FastAPI (Cerveau du projet)
├── dashboard.py          # Interface Streamlit (Monitoring)
├── mock_sensor.py        # Simulateur de capteurs (Télémétrie)
├── ml_engine/
│   ├── predictor.py      # Classe d'inférence et logique IA
│   ├── train_model.py    # Script de génération du modèle XGBoost
│   └── models/           # Dossier contenant le modèle .json
├── requirements.txt      # Liste des dépendances Python
└── README.md             # Documentation du projet
👨‍💻 Auteur
Jeff Assale - Développeur Backend & IA