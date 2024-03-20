import streamlit as st
import polars as pl
from gliner import GLiNER  # Assurez-vous que GLiNER est correctement installé

# Configuration de la page Streamlit
st.set_page_config(page_title="GliNER", page_icon="🧊", layout="wide", initial_sidebar_state="expanded")

st.title("Online NER with GliNER")
st.markdown("Prototype v0.1")

# Fonction pour charger les données depuis un fichier Excel
@st.cache_data
def load_data(file):
    data = pl.read_excel(file)
    return data

# Upload du fichier
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is None:
    st.write("Please upload a file")
    st.stop()

df = load_data(uploaded_file)  # Charge le fichier Excel téléchargé

# Sélection de la colonne pour NER
selected_column = st.selectbox("Select the column for NER:", df.columns, index=0)

# Entrée pour le filtrage et les labels NER
texte_filtrage = st.text_input("Filtrer par fonction occupée:", "")
labels_ner = st.text_input("Rentrez vos différents labels, séparés par une virgule", "")

# Filtrage basé sur le texte de filtrage, si spécifié
if texte_filtrage:
    regex = f"(?i){texte_filtrage}"
    df_filtré = df.filter(pl.col("Fonction occupée").str_contains(regex))
else:
    df_filtré = df

# Affichage du DataFrame filtré
st.dataframe(df_filtré)

# Fonction NER utilisant GLiNER
def run_ner(text, labels_list):
    model = GLiNER.from_pretrained("urchade/gliner_smallv2")
    entities = model.predict_entities(text, labels_list, threshold=0.4)
    return entities

# Bouton pour lancer le NER
if st.button("Start NER"):
    labels_list = labels_ner.split(",")
    # Préparation des séries Polars pour chaque label
    series_dict = {label: pl.Series(name=label, values=[""] * df_filtré.height) for label in labels_list}

    # Création d'une barre de progression
    progress_bar = st.progress(0)

    for index in range(df_filtré.height):
        # Mise à jour de la barre de progression
        progress_bar.progress((index + 1) / df_filtré.height)

        text_to_analyze = df_filtré.select(pl.col(selected_column))[index, 0]
        if isinstance(text_to_analyze, str):
            ner_results = run_ner(text_to_analyze, labels_list)
            for entity in ner_results:
                current_value = series_dict[entity['label']][index]
                new_value = f"{current_value}, {entity['text']}" if current_value else entity['text']
                series_dict[entity['label']][index] = new_value

    # Une fois le traitement terminé, on complète la barre de progression
    progress_bar.progress(100)

    # Ajouter les séries comme nouvelles colonnes dans df_filtré
    df_filtré = df_filtré.with_columns(list(series_dict.values()))

    # Affichage du DataFrame mis à jour
    st.dataframe(df_filtré)
