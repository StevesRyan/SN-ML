import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. PARAMÈTRES ET SCORES (A MODIFIER SELON TON NOTEBOOK)
# =============================================================================
# Regarde les résultats dans ton notebook (classification_report) et mets-les ici :
VALEUR_K = 3          
ACCURACY = 41         # Mets ici ton Accuracy (ex: 88 pour 88%)
F1_SCORE = 0.40       

st.set_page_config(
    page_title="Prédiction Santé IA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 2. DESIGN CSS "ROBUSTE" (Force le contraste Texte/Fond)
# =============================================================================
st.markdown("""
    <style>
    /* --- FORCE LE THEME CLAIR (Au cas où le config.toml n'est pas fait) --- */
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff;
        color: #000000;
    }
    [data-testid="stSidebar"] {
        background-color: #004E66; /* Bleu Nuit pour le menu */
    }

    /* --- CORRECTION DU TEXTE INVISIBLE (Force le Noir partout) --- */
    h1, h2, h3, h4, h5, h6, p, li, span, div, label {
        color: #2c3e50; /* Gris foncé presque noir */
    }
    
    /* Exception pour le texte dans la Sidebar (Blanc) */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] div {
        color: #ffffff !important;
    }

    /* --- TABLEAUX ET DATAFRAMES (Fond blanc, texte noir) --- */
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        background-color: white !important;
        color: black !important;
    }
    
    /* --- BOUTONS --- */
    div.stButton > button {
        background-color: #004E66;
        color: white !important;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #006680;
    }

    /* --- CARTES DE SCORES --- */
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        border-top: 4px solid #004E66;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #004E66 !important;
        margin: 0;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d !important;
        margin-top: 5px;
    }

    /* --- BOITES DE RESULTATS (Textes forcés en foncé) --- */
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
        font-weight: bold;
        font-size: 22px;
    }
    .success {
        background-color: #d1e7dd;
        color: #0f5132 !important; /* Vert foncé */
        border: 1px solid #badbcc;
    }
    .warning {
        background-color: #fff3cd;
        color: #856404 !important; /* Jaune foncé */
        border: 1px solid #ffeeba;
    }
    .danger {
        background-color: #f8d7da;
        color: #721c24 !important; /* Rouge foncé */
        border: 1px solid #f5c6cb;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# 3. CHARGEMENT DES DONNÉES
# =============================================================================
@st.cache_resource
def load_model_data():
    if not os.path.exists('model_knn.pkl'):
        st.error("⚠️ Fichier 'model_knn.pkl' introuvable.")
        return None
    with open('model_knn.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

@st.cache_data
def load_raw_data():
    if os.path.exists('healthcare_dataset.csv'):
        return pd.read_csv('healthcare_dataset.csv')
    return None

data_saved = load_model_data()
df_raw = load_raw_data()

# =============================================================================
# 4. NAVIGATION (SIDEBAR)
# =============================================================================
st.sidebar.markdown("<h2 style='text-align: center;'>🏥 MED-AI</h2>", unsafe_allow_html=True)

pages = ["1. Le Projet", "2. Analyse Données", "3. Performances", "4. Diagnostic"]
selection = st.sidebar.radio("Menu Principal", pages, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.info("Application de démonstration\nIngénierie 5ème Année")

# =============================================================================
# PAGE 1 : LE PROJET
# =============================================================================
if selection == "1. Le Projet":
    st.title("Système de Prédiction Médicale")
    
    st.markdown("""
    Cette application utilise l'intelligence artificielle pour assister le diagnostic médical.
    Elle se base sur l'analyse de données historiques pour prédire les anomalies.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Objectifs")
        st.info("""
        * **Rapidité** : Analyse instantanée des paramètres vitaux.
        * **Fiabilité** : Basé sur l'algorithme K-Nearest Neighbors.
        * **Support** : Aide à la décision pour les médecins.
        """)
        
    with col2:
        st.markdown("### 🛠️ Technologie")
        st.success("""
        * **Langage** : Python 3.9+
        * **Interface** : Streamlit
        * **Modèle** : Scikit-Learn (KNN)
        * **Données** : Healthcare Dataset
        """)

# =============================================================================
# PAGE 2 : ANALYSE DONNÉES
# =============================================================================
elif selection == "2. Analyse Données":
    st.title("📊 Exploration des Données")
    
    if df_raw is None:
        st.error("Dataset introuvable.")
    else:
        st.write(f"Aperçu des **{df_raw.shape[0]} dossiers patients** utilisés pour l'entraînement.")

        tab1, tab2 = st.tabs(["📄 Données Brutes", "📈 Visualisations"])
        
        with tab1:
            # On force l'affichage du dataframe
            st.dataframe(df_raw.head(15), use_container_width=True)
            st.markdown("#### Statistiques Descriptives")
            st.dataframe(df_raw.describe(), use_container_width=True)
            
        with tab2:
            colA, colB = st.columns(2)
            with colA:
                st.markdown("**Répartition des Diagnostics**")
                fig, ax = plt.subplots(facecolor='white')
                sns.countplot(x='Test Results', data=df_raw, palette="viridis", ax=ax)
                st.pyplot(fig)
            with colB:
                st.markdown("**Distribution des Âges**")
                fig2, ax2 = plt.subplots(facecolor='white')
                sns.histplot(df_raw['Age'], kde=True, color="#004E66", ax=ax2)
                st.pyplot(fig2)

# =============================================================================
# PAGE 3 : PERFORMANCES
# =============================================================================
elif selection == "3. Performances":
    st.title("🏆 Performances du Modèle")
    
    # Affichage des scores définis en haut du fichier
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Précision (Accuracy)</div><div class="metric-value">{ACCURACY}%</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">F1-Score</div><div class="metric-value">{F1_SCORE}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Voisins (K)</div><div class="metric-value">{VALEUR_K}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Matrice de Confusion (Exemple)")
    
    # Matrice fictive pour l'exemple (remplacer par image si besoin)
    cm = np.array([[45, 5], [8, 42]])
    fig, ax = plt.subplots(facecolor='white')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Anormal', 'Normal'], yticklabels=['Anormal', 'Normal'], ax=ax)
    plt.ylabel('Réalité')
    plt.xlabel('Prédiction')
    st.pyplot(fig)

# =============================================================================
# PAGE 4 : DIAGNOSTIC
# =============================================================================
elif selection == "4. Diagnostic":
    st.title("🩺 Assistant de Diagnostic")
    
    if data_saved is None:
        st.warning("Veuillez générer le modèle 'model_knn.pkl' via le Notebook.")
        st.stop()
        
    classifier = data_saved["model"]
    scaler = data_saved["scaler"]
    model_columns = data_saved["model_columns"] 
    target_mapping = data_saved.get("target_mapping", {})

    # Formulaire
    with st.sidebar.form("form_patient"):
        age = st.number_input("Âge du patient", 0, 120, 45)
        billing = st.number_input("Montant Facturé ($)", 0.0, value=1500.0)
        st.markdown("---")
        gender = st.selectbox("Genre", ["Male", "Female"])
        blood = st.selectbox("Groupe Sanguin", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
        condition = st.selectbox("Pathologie", ["Diabetes", "Hypertension", "Asthma", "Arthritis", "Cancer", "Obesity"])
        admission = st.selectbox("Type Admission", ["Emergency", "Urgent", "Elective"])
        medication = st.selectbox("Médication", ["Aspirin", "Ibuprofen", "Lipitor", "Penicillin", "Paracetamol"])
        
        btn_submit = st.form_submit_button("Lancer l'analyse")

    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.markdown("### 📝 Données Patient")
        if btn_submit:
            # Création petit tableau récapitulatif
            df_recap = pd.DataFrame({
                "Paramètre": ["Age", "Genre", "Sang", "Condition"],
                "Valeur": [age, gender, blood, condition]
            })
            st.table(df_recap)
        else:
            st.info("Remplissez le formulaire à gauche.")

    with c2:
        st.markdown("### 🔬 Résultat de l'IA")
        
        if btn_submit:
            # Encodage
            user_input = {
                'Age': age, 'Gender': gender, 'Blood Type': blood,
                'Medical Condition': condition, 'Admission Type': admission,
                'Medication': medication, 'Billing Amount': billing
            }
            # DataFrame -> OneHot -> Reindex
            input_df = pd.get_dummies(pd.DataFrame([user_input]))
            input_df = input_df.reindex(columns=model_columns, fill_value=0)
            
            # Prédiction
            X_scaled = scaler.transform(input_df)
            pred_idx = classifier.predict(X_scaled)[0]
            pred_proba = classifier.predict_proba(X_scaled)[0]
            
            # Décodage (0/1 -> Texte)
            inv_map = {v: k for k, v in target_mapping.items()}
            res_text = inv_map.get(pred_idx, "Inconnu")
            confiance = np.max(pred_proba) * 100
            
            # Affichage Couleur
            if res_text == "Normal":
                st.markdown(f'<div class="result-box success">✅ Résultat : {res_text}</div>', unsafe_allow_html=True)
                st.caption("Le profil correspond à un dossier standard.")
            elif res_text == "Inconclusive":
                st.markdown(f'<div class="result-box warning">⚠️ Résultat : {res_text}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box danger">🚨 Résultat : {res_text}</div>', unsafe_allow_html=True)
                st.caption("Le modèle a détecté des indicateurs atypiques.")
            
            st.markdown(f"**Indice de Confiance :** {confiance:.2f}%")
            
            # Graphe
            probs = pd.DataFrame(pred_proba.reshape(1, -1), columns=[inv_map[i] for i in range(len(inv_map))])
            st.bar_chart(probs.T)