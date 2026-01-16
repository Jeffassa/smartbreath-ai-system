import psycopg2
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from dotenv import load_dotenv

load_dotenv()

def get_performance_stats():
    """Analyse la précision de l'IA en comparant status et actual_outcome"""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        
        # On récupère uniquement les données où le patient a donné un feedback
        query = """
            SELECT status, actual_outcome 
            FROM sensor_data 
            WHERE actual_outcome IS NOT NULL
        """
        df = pd.read_sql(query, conn)
        conn.close()

        if df.empty:
            print(" Aucune donnée de feedback disponible pour le calcul.")
            return

        # Transformation en binaire pour comparaison
        # IA positive si Status est PREVENTION ou CRITIQUE
        df['pred_binary'] = df['status'].apply(lambda x: 1 if x in ['PRÉVENTION', 'CRITIQUE'] else 0)
        df['actual_binary'] = df['actual_outcome'].astype(int)

        # Calcul des métriques
        acc = accuracy_score(df['actual_binary'], df['pred_binary'])
        cm = confusion_matrix(df['actual_binary'], df['pred_binary'])
        
        # --- AFFICHAGE DES RÉSULTATS ---
        print("\n" + "="*40)
        print(f"RAPPORT DE PERFORMANCE IA - {pd.Timestamp.now().strftime('%Y-%m-%d')}")
        print("="*40)
        print(f" Taux de précision global : {acc*100:.2f}%")
        print(f" Total feedbacks analysés : {len(df)}")
        print("-" * 40)
        
        # Détails de la Matrice de Confusion
        tn, fp, fn, tp = cm.ravel()
        print(f" Vrais Positifs (Crises détectées) : {tp}")
        print(f" Faux Positifs (Fausses alertes)   : {fp}")
        print(f"Vrais Négatifs (Stabilité confirmée): {tn}")
        print(f" Faux Négatifs (Crises manquées)   : {fn}")
        print("-" * 40)

        # --- GÉNÉRATION D'UN GRAPHIQUE ---
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Prédit Stable', 'Prédit Crise'],
                    yticklabels=['Réel Stable', 'Réel Crise'])
        plt.title('Matrice de Confusion : Prédit vs Réel')
        plt.ylabel('Réalité (Patient)')
        plt.xlabel('Prédiction (IA)')
        
        # Sauvegarde du rapport visuel
        plt.savefig('ia_performance_report.png')
        print(" Graphique 'ia_performance_report.png' généré.")

    except Exception as e:
        print(f" Erreur lors de la maintenance : {e}")

if __name__ == "__main__":
    get_performance_stats()