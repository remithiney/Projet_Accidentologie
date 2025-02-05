import streamlit as st
import joblib
import pandas as pd
import os
import json
import xgboost
import lightgbm
import matplotlib.pyplot as plt

# Chemin du dossier contenant les modèles
MODEL_DIR = "../models/"
MODELS = ["xgboost_0.6.joblib", "random_forest_0.6.joblib"]

current_path = os.getcwd()
print(f"Le chemin courant est : {current_path}")

# Fonction pour charger un modèle
@st.cache_resource
def load_model(model_name):
    model_path = os.path.join(MODEL_DIR, model_name)
    return joblib.load(model_path)

# Fonction pour créer un widget en fonction des caractéristiques de la variable
def create_widget(var_name, var_info):
    if var_info["type"] == "numerical":
        min_val, max_val = var_info["values"]
        if min_val == 0 and max_val == 1:
            return st.checkbox(var_name, value=False, key=var_name)
        else:
            return st.slider(var_name, min_val, max_val, value=(min_val + max_val) // 2, key=var_name)
    elif var_info["type"] == "categorial":
        return st.selectbox(var_name, var_info["values"], key=var_name)

# Charger le fichier JSON des variables
with open("./streamlit/output.json", "r") as file:
    variables = json.load(file)

# Liste des colonnes dans l'ordre requis
required_columns = ['place', 'catu', 'sexe', 'trajet', 'locp', 'actp', 'catv', 'obs',
                      'obsm', 'manv', 'jour', 'mois', 'lum', 'agg', 'int', 'atm', 'col',
                      'catr', 'circ', 'vosp', 'prof', 'plan', 'surf', 'infra', 'situ', 'vma',
                      'nb_v', 'nb_u', 'imply_cycle_edp', 'imply_2rm', 'imply_vl', 'imply_pl',
                      'age', 'jour_semaine', 'hr', 'reg', 'has_ceinture', 'has_gants',
                      'has_casque', 'has_airbag', 'has_gilet', 'has_de', 'choc_avant',
                      'choc_arriere', 'choc_gauche', 'choc_droit']

# Interface utilisateur
st.title("Formulaire de saisie des données")

# Sélection du modèle
selected_model = st.selectbox(
    "Choisissez un modèle pour la prédiction",
    options=MODELS,
    index=0
)

# Charger le modèle sélectionné
model_pipeline = load_model(selected_model)

# Formulaire de saisie des données utilisateur
user_inputs = {}
for group_name in ["contexte", "vehicule", "usager"]:
    st.header(group_name.capitalize())
    group_vars = {k: v for k, v in variables.items() if v["group"] == group_name}
    columns = st.columns(3)
    col_index = 0
    for var_name, var_info in group_vars.items():
        with columns[col_index]:
            user_inputs[var_name] = create_widget(var_name, var_info)
        col_index = (col_index + 1) % 3

# Le seuil de décision est défini ici, avant la soumission du formulaire
st.subheader("Paramètre de prédiction")
seuil = st.slider("Choisissez le seuil de décision :", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Bouton de soumission
if st.button("Soumettre"):
    # Organiser les données dans l'ordre requis
    final_data = {}
    for col in required_columns:
        if col in user_inputs:
            final_data[col] = (
                int(user_inputs[col]) if isinstance(user_inputs[col], bool)
                else user_inputs[col]
            )
        else:
            # Valeurs par défaut si la variable n'est pas renseignée
            var_info = variables.get(col, {})
            if var_info.get("type") == "numerical":
                min_val, max_val = var_info.get("values", [0, 1])
                final_data[col] = (min_val + max_val) // 2
            elif var_info.get("type") == "categorial":
                final_data[col] = var_info.get("values", [""])[0]

    # Convertir les données en DataFrame pour la prédiction
    input_df = pd.DataFrame([final_data])
    st.write("Données saisies :", input_df)

    # Calculer la probabilité de la classe 1
    proba = model_pipeline.predict_proba(input_df)[:, 1][0]
    prediction = 1 if proba >= seuil else 0

    result_html = f"""
    <div style="
        background-color: #e6f7ff;
        border: 2px solid #1890ff;
        border-radius: 5px;
        padding: 15px;
        margin-top: 20px;
    ">
        <h2 style="color: #1890ff;">Résultats de la Prédiction</h2>
        <p style="font-size: 18px;"><strong>Probabilité prédite pour la classe 1 :</strong> {proba:.2f}</p>
        <p style="font-size: 18px;"><strong>Seuil choisi :</strong> {seuil:.2f}</p>
        <p style="font-size: 18px;"><strong>Prédiction binaire :</strong> {prediction}</p>
        <p style="font-size: 16px; color: #595959;">
            Une probabilité élevée (proche de 1) indique une forte propension à la classe 1.<br>
            Ajustez le seuil de décision pour observer comment la prédiction binaire évolue.
        </p>
    </div>
    """

    st.markdown(result_html, unsafe_allow_html=True)

    # Affichage additionnel via les métriques Streamlit
    col1, col2 = st.columns(2)
    col1.metric("Probabilité", f"{proba:.2f}")
    col2.metric("Prédiction", prediction)
