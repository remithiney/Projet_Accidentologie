import streamlit as st
import joblib
import pandas as pd
import os
import json
import shap
import xgboost
import lightgbm
import matplotlib.pyplot as plt

# Chemin du dossier contenant les modèles
MODEL_DIR = "./models/joblib/"
MODELS = ["RandomForest-roc_auc.joblib","LightGBM-roc_auc.joblib","XGBoost-roc_auc.joblib"]

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

# Formulaire de saisie
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

# Soumission du formulaire
if st.button("Soumettre"):
    # Organiser les données dans l'ordre requis
    final_data = {}
    for col in required_columns:
        if col in user_inputs:
            final_data[col] = (
                int(user_inputs[col]) if isinstance(user_inputs[col], bool)  # Convertir checkbox en 0/1
                else user_inputs[col]
            )
        else:
            # Valeurs par défaut
            var_info = variables.get(col, {})
            if var_info.get("type") == "numerical":
                min_val, max_val = var_info.get("values", [0, 1])
                final_data[col] = (min_val + max_val) // 2
            elif var_info.get("type") == "categorial":
                final_data[col] = var_info.get("values", [""])[0]

    # Convertir les données en DataFrame pour la prédiction
    input_df = pd.DataFrame([final_data])
    try:
        # Prédiction avec les données utilisateur
        prediction = model_pipeline.predict(input_df)
        predicted_class = int(prediction[0])  # Classe prédite
        prediction_proba = model_pipeline.predict_proba(input_df)[0]  # Probabilités des classes


        assurance = prediction_proba[predicted_class]  # Problème binaire : probabilité de la classe prédite

        st.success(f"Prédiction : Classe {predicted_class}")
        st.info(f"Assurance dans la prédiction : {assurance:.4f}")

        # Calcul des valeurs SHAP pour les données utilisateur
        model = model_pipeline.named_steps['classifier']
        preprocessor = model_pipeline.named_steps['preprocessor']

        # Transformer les données utilisateur
        X_transformed = preprocessor.transform(input_df)
        feature_names = preprocessor.get_feature_names_out()  # Obtenir les noms des colonnes après transformation
        X_df = pd.DataFrame(X_transformed, columns=feature_names)

        # Initialiser l'explainer SHAP
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer(X_df)

        # Vérification de la dimension de shap_values
        if len(shap_values.values.shape) == 2:
            # Cas 2D : tableau de (observations, caractéristiques)
            shap_values_for_class = shap_values.values[0, :]
            base_value_for_class = shap_values.base_values[0]
        elif len(shap_values.values.shape) == 3:
            # Cas 3D : tableau de (observations, caractéristiques, classes)
            shap_values_for_class = shap_values.values[0, :, 1]
            base_value_for_class = shap_values.base_values[0, 1]
        else:
            raise ValueError("Structure inattendue pour les valeurs SHAP.")

        # Données pour la première observation
        data_for_observation = shap_values.data[0]

        # Générer un waterfall plot pour expliquer la prédiction
        st.subheader("Pourquoi le modèle a fait cette prédiction ?")
        shap.initjs()

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_for_class,
                base_values=base_value_for_class,
                feature_names=feature_names,
                data=data_for_observation
            ),
            max_display=50
        )
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erreur lors du calcul ou de l'affichage du waterfall plot : {e}")





