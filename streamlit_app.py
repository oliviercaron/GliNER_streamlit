import os  # Add this import to use os.path.splitext
import csv
import streamlit as st
import polars as pl
from io import BytesIO, StringIO
from gliner import GLiNER
from gliner_file import run_ner
import time

st.set_page_config(page_title="GliNER", page_icon="🧊", layout="wide", initial_sidebar_state="expanded")

# Modified function to load data from either an Excel or CSV file
@st.cache_data
def load_data(file):
    _, file_ext = os.path.splitext(file.name)
    if file_ext.lower() in ['.xls', '.xlsx']:
        return pl.read_excel(file)
    elif file_ext.lower() == '.csv':
        file.seek(0)  # Retour au début du fichier
        try:
            sample = file.read(4096).decode('utf-8')  # Essayer de décoder l'échantillon en UTF-8
            encoding = 'utf-8'
        except UnicodeDecodeError:
            encoding = 'latin1'  # Basculer sur 'latin1' si UTF-8 échoue
            file.seek(0)
            sample = file.read(4096).decode(encoding)
        
        file.seek(0)
        dialect = csv.Sniffer().sniff(sample)  # Détecter le dialecte/délimiteur

        # Convertir le fichier en StringIO pour simuler un fichier texte, si nécessaire
        file.seek(0)
        if encoding != 'utf-8':
            file_content = file.read().decode(encoding)
            file = StringIO(file_content)
        else:
            file_content = file.read().decode('utf-8')
            file = StringIO(file_content)
        
        return pl.read_csv(file, separator=dialect.delimiter, truncate_ragged_lines=True, ignore_errors=True)
    else:
        raise ValueError("The uploaded file must be a CSV or Excel file.")


# Function to perform NER and update the UI
def perform_ner(filtered_df, selected_column, labels_list):
    ner_results_dict = {label: [] for label in labels_list}
    
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    start_time = time.time()  # Enregistrer le temps de début pour le temps d'exécution total

    for index, row in enumerate(filtered_df.to_pandas().itertuples(), 1):
        iteration_start_time = time.time()  # Temps de début pour cette itération
        
        if st.session_state.stop_processing:
            progress_text.text("Process stopped by the user.")
            break

        text_to_analyze = getattr(row, selected_column)
        ner_results = run_ner(st.session_state.gliner_model, text_to_analyze, labels_list)

        for label in labels_list:
            texts = ner_results.get(label, [])
            concatenated_texts = ', '.join(texts)
            ner_results_dict[label].append(concatenated_texts)

        progress = index / filtered_df.height
        progress_bar.progress(progress)
        
        iteration_time = time.time() - iteration_start_time  # Calculer le temps d'exécution pour cette itération
        total_time = time.time() - start_time  # Calculer le temps total écoulé jusqu'à présent
        
        progress_text.text(f"Progress: {index}/{filtered_df.height} - {progress * 100:.0f}% (Iteration: {iteration_time:.2f}s, Total: {total_time:.2f}s)")

    end_time = time.time()  # Enregistrer le temps de fin
    total_execution_time = end_time - start_time  # Calculer le temps d'exécution total
    
    progress_text.text(f"Processing complete! Total execution time: {total_execution_time:.2f}s")
    
    for label, texts in ner_results_dict.items():
        filtered_df = filtered_df.with_columns(pl.Series(name=label, values=texts))

    return filtered_df

def main():
    st.title("Online NER with GliNER")
    st.markdown("Prototype v0.1")

    # Ensure the stop_processing flag is initialized
    if 'stop_processing' not in st.session_state:
        st.session_state.stop_processing = False

    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is None:
        st.warning("Please upload a file.")
        return

    try:
        df = load_data(uploaded_file)
    except ValueError as e:
        st.error(str(e))
        return

    selected_column = st.selectbox("Select the column for NER:", df.columns, index=0)
    filter_text = st.text_input("Filter column by input text", "")
    ner_labels = st.text_input("Enter all your different labels, separated by a comma", "")

    filtered_df = df.filter(pl.col(selected_column).str.contains(f"(?i).*{filter_text}.*")) if filter_text else df
    st.dataframe(filtered_df)

    if st.button("Start NER"):
        if not ner_labels:
            st.warning("Please enter some labels for NER.")
        else:
            # Load GLiNER model if not already loaded
            if 'gliner_model' not in st.session_state:
                with st.spinner('Loading GLiNER model... Please wait.'):
                    st.session_state.gliner_model = GLiNER.from_pretrained("urchade/gliner_largev2")
                    st.session_state.gliner_model.eval()
                    
            labels_list = ner_labels.split(",")
            updated_df = perform_ner(filtered_df, selected_column, labels_list)
            st.dataframe(updated_df)

            def to_excel(df):
                output = BytesIO()
                df.to_pandas().to_excel(output, index=False, engine='openpyxl')
                return output.getvalue()

            df_excel = to_excel(updated_df)
            st.download_button(label="📥 Download Excel",
                               data=df_excel,
                               file_name="ner_results.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.button("Stop Processing", on_click=lambda: setattr(st.session_state, 'stop_processing', True))

if __name__ == "__main__":
    main()
