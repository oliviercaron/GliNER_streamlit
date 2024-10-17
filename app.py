import os
import csv
import streamlit as st
import polars as pl
from io import BytesIO, StringIO
from gliner import GLiNER
from gliner_file import run_ner
import time
import torch
import platform
from typing import List
from streamlit_tags import st_tags  # Import du composant st_tags

# Configuration de la page Streamlit
st.set_page_config(
    page_title="GLiNER",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour charger les données du fichier téléchargé
@st.cache_data
def load_data(file):
    """
    Charge un fichier CSV ou Excel téléchargé avec une détection résiliente des délimiteurs et des types.
    """
    # Message de chargement personnalisé
    with st.spinner("Chargement des données, veuillez patienter..."):
        try:
            _, file_ext = os.path.splitext(file.name)

            if file_ext.lower() in [".xls", ".xlsx"]:
                return load_excel(file)
            elif file_ext.lower() == ".csv":
                return load_csv(file)
            else:
                raise ValueError("Format de fichier non pris en charge. Veuillez télécharger un fichier CSV ou Excel.")
        except Exception as e:
            st.error("Erreur lors du chargement des données :")
            st.error(str(e))
            return None

def load_excel(file):
    """
    Gère le chargement des fichiers Excel en mode tolérant aux erreurs.
    """
    try:
        # Lire le fichier Excel avec Polars et ignorer les erreurs de parsing.
        # Utilise Pandas comme alternative si des problèmes surviennent.
        try:
            df = pl.read_excel(file, read_options={"ignore_errors": True})
        except Exception:
            st.warning("Échec de chargement avec Polars. Tentative avec Pandas.")
            # Chargement avec Pandas puis conversion vers Polars
            df = pd.read_excel(file, engine="openpyxl")
            df = pl.from_pandas(df)
        return df
    except Exception as e:
        raise ValueError(f"Erreur lors de la lecture du fichier Excel : {str(e)}")

def load_csv(file):
    """
    Gère le chargement des fichiers CSV avec détection automatique des délimiteurs et tolérance aux erreurs.
    """
    try:
        file.seek(0)
        raw_data = file.read()
        try:
            file_content = raw_data.decode('utf-8')
        except UnicodeDecodeError:
            try:
                file_content = raw_data.decode('latin1')
            except UnicodeDecodeError:
                raise ValueError("Impossible de décoder le fichier. Assurez-vous qu'il est encodé en UTF-8 ou Latin-1.")
        
        # Détection améliorée du délimiteur avec test de cohérence
        sample = file_content[:4096]
        delimiters = [",", ";", "|", "\t"]
        delimiter = detect_delimiter(sample, delimiters)
        
        # Chargement complet avec le délimiteur détecté
        df = pl.read_csv(
            StringIO(file_content),
            separator=delimiter,
            try_parse_dates=True,
            ignore_errors=True,  # Ignorer les erreurs pour les valeurs incorrectes
            truncate_ragged_lines=True
        )
        return df
    except Exception as e:
        raise ValueError(f"Erreur lors de la lecture du fichier CSV : {str(e)}")

def detect_delimiter(sample, delimiters):
    """
    Détecte un délimiteur valide dans un échantillon de texte en testant chaque délimiteur courant.
    """
    for delim in delimiters:
        try:
            temp_df = pl.read_csv(StringIO(sample), separator=delim, n_rows=10)
            # Vérifier que le nombre de colonnes est cohérent dans l'échantillon
            if len(set(len(row) for row in temp_df.rows())) == 1:
                return delim
        except Exception:
            continue
    raise ValueError("Impossible de détecter un délimiteur cohérent. Veuillez vérifier le format du fichier.")

# Fonction pour charger le modèle GLiNER
@st.cache_resource
def load_model():
    """
    Charge le modèle GLiNER en mémoire pour éviter les rechargements multiples.
    """
    try:
        gpu_available = torch.cuda.is_available()

        with st.spinner("Chargement du modèle GLiNER... Veuillez patienter."):
            device = torch.device("cuda" if gpu_available else "cpu")
            model = GLiNER.from_pretrained(
                "urchade/gliner_multi-v2.1"
            ).to(device)
            model.eval()

        if gpu_available:
            device_name = torch.cuda.get_device_name(0)
            st.success(f"GPU détecté : {device_name}. Modèle chargé sur GPU.")
        else:
            cpu_name = platform.processor()
            st.warning(f"GPU non détecté. Utilisation du CPU : {cpu_name}")

        return model
    except Exception as e:
        st.error("Erreur lors du chargement du modèle :")
        st.error(str(e))
        return None

# Fonction pour effectuer le NER et mettre à jour l'interface utilisateur
def perform_ner(filtered_df, selected_column, labels_list, threshold):
    """
    Exécute la reconnaissance d'entités nommées (NER) sur les données filtrées.
    """
    try:
        texts_to_analyze = filtered_df[selected_column].to_list()
        total_rows = len(texts_to_analyze)
        ner_results_list = []

        # Initialisation de la barre de progression et du texte
        progress_bar = st.progress(0)
        progress_text = st.empty()
        start_time = time.time()

        # Traitement de chaque ligne individuellement pour garder les mises à jour de progression réactives
        for index, text in enumerate(texts_to_analyze, 1):
            if st.session_state.stop_processing:
                progress_text.text("Traitement arrêté par l'utilisateur.")
                break

            ner_results = run_ner(
                st.session_state.gliner_model,
                [text],
                labels_list,
                threshold=threshold
            )
            ner_results_list.append(ner_results)

            # Mise à jour de la barre de progression et du texte après chaque ligne
            progress = index / total_rows
            elapsed_time = time.time() - start_time
            progress_bar.progress(progress)
            progress_text.text(f"Progression : {index}/{total_rows} - {progress * 100:.0f}% (Temps écoulé : {elapsed_time:.2f}s)")

        # Ajout des résultats NER au DataFrame
        for label in labels_list:
            extracted_entities = []
            for entities in ner_results_list:
                texts = [entity["text"] for entity in entities[0] if entity["label"] == label]
                concatenated_texts = ", ".join(texts) if texts else ""
                extracted_entities.append(concatenated_texts)
            filtered_df = filtered_df.with_columns(pl.Series(name=label, values=extracted_entities))

        end_time = time.time()
        st.success(f"Traitement terminé en {end_time - start_time:.2f} secondes.")

        return filtered_df
    except Exception as e:
        st.error(f"Erreur lors du traitement NER : {str(e)}")
        return filtered_df

# Fonction principale pour exécuter l'application Streamlit
def main():
    st.title("Reconnaissance d'Entités Nommées en Ligne avec GLiNER")
    st.markdown("Prototype v0.1")

    # Instructions pour l'utilisateur
    st.write("""
    Cette application effectue la reconnaissance d'entités nommées (NER) sur vos données textuelles en utilisant GLiNER.

    **Instructions :**
    1. Téléchargez un fichier CSV ou Excel.
    2. Sélectionnez la colonne contenant le texte à analyser.
    3. Filtrez les données si nécessaire.
    4. Entrez les labels NER que vous souhaitez détecter.
    5. Cliquez sur "Démarrer le NER" pour commencer le traitement.
    """)

    # Initialisation des variables de session
    if "stop_processing" not in st.session_state:
        st.session_state.stop_processing = False
    if "threshold" not in st.session_state:
        st.session_state.threshold = 0.4
    if "labels_list" not in st.session_state:
        st.session_state.labels_list = []

    # Chargement du modèle
    st.session_state.gliner_model = load_model()
    if st.session_state.gliner_model is None:
        return

    # Téléchargement du fichier
    uploaded_file = st.sidebar.file_uploader("Choisissez un fichier (CSV ou Excel)")
    if uploaded_file is None:
        st.warning("Veuillez télécharger un fichier pour continuer.")
        return

    # Chargement des données
    df = load_data(uploaded_file)
    if df is None:
        return

    # Sélection de la colonne
    selected_column = st.selectbox("Sélectionnez la colonne contenant le texte :", df.columns)

    # Filtrage des données
    filter_text = st.text_input("Filtrer la colonne par texte", "")
    if filter_text:
        filtered_df = df.filter(pl.col(selected_column).str.contains(f"(?i).*{filter_text}.*"))
    else:
        filtered_df = df

    st.write("**Aperçu des données filtrées :**")

    # Définir le nombre de lignes par page
    rows_per_page = 100
    total_rows = len(filtered_df)
    total_pages = (total_rows // rows_per_page) + (1 if total_rows % rows_per_page != 0 else 0)
    
    # Initialiser l'état de la page dans session_state
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    
    # Afficher les lignes de la page courante
    start_idx = (st.session_state.current_page - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, total_rows)
    st.dataframe(filtered_df.slice(start_idx, end_idx - start_idx).to_dicts(), use_container_width=True)

    # Navigation de pagination
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("⏮️ Première"):
            st.session_state.current_page = 1
    with col2:
        if st.button("⬅️ Précédente"):
            if st.session_state.current_page > 1:
                st.session_state.current_page -= 1
    with col3:
        st.write(f"Page {st.session_state.current_page} sur {total_pages}")
    with col4:
        if st.button("Suivante ➡️"):
            if st.session_state.current_page < total_pages:
                st.session_state.current_page += 1
    with col5:
        if st.button("Dernière ⏭️"):
            st.session_state.current_page = total_pages

    # Entrée dynamique des labels NER avec st_tags
    st.write("**Entrez les labels NER :**")
    st.session_state.labels_list = st_tags(
        label='',
        text='Appuyez sur Entrée pour ajouter un label',
        value=st.session_state.get('labels_list', []),
        suggestions=[],
        maxtags=10,
        key='labels',
    )

    # Curseur pour le seuil de confiance
    st.slider("Définissez le seuil de confiance", 0.0, 1.0, st.session_state.threshold, 0.01, key="threshold")

    # Boutons pour démarrer et arrêter le NER
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Démarrer le NER")
    with col2:
        stop_button = st.button("Arrêter")

    if start_button:
        st.session_state.stop_processing = False

        if not st.session_state.labels_list:
            st.warning("Veuillez entrer des labels pour le NER.")
        else:
            # Exécuter le NER
            updated_df = perform_ner(filtered_df, selected_column, st.session_state.labels_list, st.session_state.threshold)
            st.write("**Résultats du NER :**")
            st.dataframe(updated_df.to_pandas(), use_container_width=True)

            # Fonction pour convertir le DataFrame en Excel
            def to_excel(df):
                output = BytesIO()
                df.write_excel(output)
                return output.getvalue()

            # Fonction pour convertir le DataFrame en CSV
            def to_csv(df):
                return df.write_csv().encode('utf-8')

            # Boutons de téléchargement des résultats
            download_col1, download_col2 = st.columns(2)
            with download_col1:
                st.download_button(
                    label="📥 Télécharger en Excel",
                    data=to_excel(updated_df),
                    file_name="resultats_ner.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            with download_col2:
                st.download_button(
                    label="📥 Télécharger en CSV",
                    data=to_csv(updated_df),
                    file_name="resultats_ner.csv",
                    mime="text/csv",
                )

    if stop_button:
        st.session_state.stop_processing = True
        st.warning("Traitement arrêté par l'utilisateur.")

if __name__ == "__main__":
    main()
