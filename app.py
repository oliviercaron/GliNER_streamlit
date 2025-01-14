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
from streamlit_tags import st_tags  # Importing the st_tags component
from transformers import AutoTokenizer

# Streamlit page configuration
st.set_page_config(
    page_title="GLiNER",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- List of official GLiNER models (you can adapt or extend this list) ---
OFFICIAL_MODELS = [
    "urchade/gliner_base",         # v0
    "urchade/gliner_multi",        # v0
    "urchade/gliner_small-v1",     # v1
    "urchade/gliner_medium-v1",    # v1
    "urchade/gliner_large-v1",     # v1
    "urchade/gliner_small-v2",     # v2
    "urchade/gliner_medium-v2",    # v2
    "urchade/gliner_large-v2",     # v2
    "urchade/gliner_small-v2.1",   # v2.1
    "urchade/gliner_medium-v2.1",  # v2.1
    "urchade/gliner_large-v2.1",   # v2.1
    "urchade/gliner_multi-v2.1"    # v2.1
]

# --- Function to load a custom fine-tuned model from a local path ---
def load_custom_model(model_path):
    # Comments in English: This function loads a locally fine-tuned model.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    model = GLiNER.from_pretrained(model_path, load_tokenizer=False).to(device)
    model.data_processor.transformer_tokenizer = tokenizer
    return model

# --- Function to discover custom models inside the "models" folder ---
def get_custom_models_list(models_root="models"):
    # Comments in English: This function returns the subdirectories inside the 'models' folder as potential custom models.
    if not os.path.isdir(models_root):
        return []
    return [
        d for d in os.listdir(models_root)
        if os.path.isdir(os.path.join(models_root, d))
    ]

# Function to load data from the uploaded file
@st.cache_data
def load_data(file):
    """
    Loads an uploaded CSV or Excel file with resilient detection of delimiters and types.
    """
    with st.spinner("Loading data, please wait..."):
        try:
            _, file_ext = os.path.splitext(file.name)
            if file_ext.lower() in [".xls", ".xlsx"]:
                return load_excel(file)
            elif file_ext.lower() == ".csv":
                return load_csv(file)
            else:
                raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
        except Exception as e:
            st.error("Error loading data:")
            st.error(str(e))
            return None

def load_excel(file):
    """
    Loads an Excel file using `BytesIO` and `polars` for reduced latency.
    """
    try:
        file_bytes = BytesIO(file.read())
        df = pl.read_excel(file_bytes, read_options={"ignore_errors": True})
        return df
    except Exception as e:
        raise ValueError(f"Error reading the Excel file: {str(e)}")

def load_csv(file):
    """
    Loads a CSV file by detecting the delimiter and using the quote character to handle internal delimiters.
    """
    try:
        file.seek(0)  # Reset file pointer to ensure reading from the beginning
        raw_data = file.read()

        try:
            file_content = raw_data.decode('utf-8')
        except UnicodeDecodeError:
            try:
                file_content = raw_data.decode('latin1')
            except UnicodeDecodeError:
                raise ValueError("Unable to decode the file. Ensure it is encoded in UTF-8 or Latin-1.")
        
        delimiters = [",", ";", "|", "\t", " "]

        for delimiter in delimiters:
            try:
                df = pl.read_csv(
                    StringIO(file_content),
                    separator=delimiter,
                    quote_char='"',
                    try_parse_dates=True,
                    ignore_errors=True,
                    truncate_ragged_lines=True
                )
                return df
            except Exception:
                continue

        raise ValueError("Unable to load the file with common delimiters.")
    except Exception as e:
        raise ValueError(f"Error reading the CSV file: {str(e)}")

# --- Main model loader (handles both official and custom models) ---
@st.cache_resource
def load_model(model_name):
    """
    Loads the GLiNER model into memory to avoid multiple reloads.
    This function handles both official and custom models.
    """
    try:
        gpu_available = torch.cuda.is_available()
        device = torch.device("cuda" if gpu_available else "cpu")

        # Comments in English: If model_name is in OFFICIAL_MODELS, we load directly from Hugging Face.
        # Otherwise, we assume it's a custom model in the "models" folder.
        if model_name in OFFICIAL_MODELS:
            with st.spinner(f"Loading official model: {model_name} ... Please wait."):
                model = GLiNER.from_pretrained(model_name).to(device)
        else:
            # Comments in English: For custom model, the local path is "models/<model_name>" by default.
            custom_model_path = os.path.join("models", model_name)
            with st.spinner(f"Loading custom model from {custom_model_path} ... Please wait."):
                model = load_custom_model(custom_model_path)

        model.eval()

        if gpu_available:
            device_name = torch.cuda.get_device_name(0)
            st.success(f"GPU detected: {device_name}. Model loaded on GPU.")
        else:
            cpu_name = platform.processor()
            st.warning(f"No GPU detected. Using CPU: {cpu_name}")

        return model

    except Exception as e:
        st.error("Error loading the model:")
        st.error(str(e))
        return None

def perform_ner(filtered_df, selected_column, labels_list, threshold):
    """
    Executes named entity recognition (NER) on the filtered data.
    """
    try:
        texts_to_analyze = filtered_df[selected_column].to_list()
        total_rows = len(texts_to_analyze)
        ner_results_list = []

        progress_bar = st.progress(0)
        progress_text = st.empty()
        start_time = time.time()

        for index, text in enumerate(texts_to_analyze, 1):
            if st.session_state.stop_processing:
                progress_text.text("Processing stopped by user.")
                break

            ner_results = run_ner(
                st.session_state.gliner_model,
                [text],
                labels_list,
                threshold=threshold
            )
            ner_results_list.append(ner_results)

            progress = index / total_rows
            elapsed_time = time.time() - start_time
            progress_bar.progress(progress)
            progress_text.text(f"Progress: {index}/{total_rows} - {progress * 100:.0f}% (Elapsed time: {elapsed_time:.2f}s)")

        # Comments in English: After collecting all NER results, we add one column per label.
        for label in labels_list:
            extracted_entities = []
            for entities in ner_results_list:
                texts = [entity["text"] for entity in entities[0] if entity["label"] == label]
                concatenated_texts = ", ".join(texts) if texts else ""
                extracted_entities.append(concatenated_texts)
            filtered_df = filtered_df.with_columns(pl.Series(name=label, values=extracted_entities))

        end_time = time.time()
        st.success(f"Processing completed in {end_time - start_time:.2f} seconds.")

        return filtered_df
    except Exception as e:
        st.error(f"Error during NER processing: {str(e)}")
        return filtered_df

def main():
    st.title("Use NER with GliNER on your data file")
    st.markdown("Prototype v0.1")

    st.write("""
    This application performs named entity recognition (NER) on your text data using GLiNER.

    **Instructions:**
    1. Upload a CSV or Excel file.
    2. Select the column containing the text to analyze.
    3. Filter the data if necessary.
    4. Enter the NER labels you wish to detect.
    5. Click "Start NER" to begin processing.
    """)

    if "stop_processing" not in st.session_state:
        st.session_state.stop_processing = False
    if "threshold" not in st.session_state:
        st.session_state.threshold = 0.4
    if "labels_list" not in st.session_state:
        st.session_state.labels_list = []

    # -- Model selection in the sidebar --
    st.sidebar.write("## Model Selection")
    custom_models = get_custom_models_list("models")
    all_models_choices = OFFICIAL_MODELS + custom_models

    # Comments in English: If you wish to set a specific default model, you can change the 'index' parameter.
    selected_model = st.sidebar.selectbox(
        "Select your GLiNER model:",
        all_models_choices,
        index=all_models_choices.index("urchade/gliner_multi-v2.1") if "urchade/gliner_multi-v2.1" in all_models_choices else 0
    )

    st.session_state.gliner_model = load_model(selected_model)
    if st.session_state.gliner_model is None:
        return

    uploaded_file = st.sidebar.file_uploader("Choose a file (CSV or Excel)")
    if uploaded_file is None:
        st.warning("Please upload a file to continue.")
        return

    df = load_data(uploaded_file)
    if df is None:
        return

    selected_column = st.selectbox("Select the column containing the text:", df.columns)

    filter_text = st.text_input("Filter the column by text", "")
    if filter_text:
        filtered_df = df.filter(pl.col(selected_column).str.contains(f"(?i).*{filter_text}.*"))
    else:
        filtered_df = df

    st.write("Filtered data preview:")

    rows_per_page = 100
    total_rows = len(filtered_df)
    total_pages = (total_rows - 1) // rows_per_page + 1

    if "current_page" not in st.session_state:
        st.session_state.current_page = 1

    def update_page(new_page):
        st.session_state.current_page = new_page

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        first = st.button("⏮️ First")
    with col2:
        previous = st.button("⬅️ Previous")
    with col3:
        pass
    with col4:
        next_btn = st.button("Next ➡️")
    with col5:
        last = st.button("Last ⏭️")

    if first:
        update_page(1)
    elif previous:
        if st.session_state.current_page > 1:
            update_page(st.session_state.current_page - 1)
    elif next_btn:
        if st.session_state.current_page < total_pages:
            update_page(st.session_state.current_page + 1)
    elif last:
        update_page(total_pages)

    with col3:
        st.markdown(f"Page **{st.session_state.current_page}** of **{total_pages}**")

    start_idx = (st.session_state.current_page - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, total_rows)

    if not filtered_df.is_empty():
        current_page_data = filtered_df.slice(start_idx, end_idx - start_idx)
        st.write(f"Displaying {start_idx + 1} to {end_idx} of {total_rows} rows")
        st.dataframe(current_page_data.to_pandas(), use_container_width=True)
    else:
        st.warning("The filtered DataFrame is empty. Please check your filters.")

    st.slider("Set confidence threshold", 0.0, 1.0, st.session_state.threshold, 0.01, key="threshold")

    st.session_state.labels_list = st_tags(
        label="Enter the NER labels to detect",
        text="Add more labels as needed",
        value=st.session_state.labels_list,
        key="1"
    )

    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Start NER")
    with col2:
        stop_button = st.button("Stop")

    if start_button:
        st.session_state.stop_processing = False

        if not st.session_state.labels_list:
            st.warning("Please enter labels for NER.")
        else:
            updated_df = perform_ner(filtered_df, selected_column, st.session_state.labels_list, st.session_state.threshold)
            st.write("**NER Results:**")
            st.dataframe(updated_df.to_pandas(), use_container_width=True)

            def to_excel(df):
                output = BytesIO()
                df.write_excel(output)
                return output.getvalue()

            def to_csv(df):
                return df.write_csv().encode('utf-8')

            download_col1, download_col2 = st.columns(2)
            with download_col1:
                st.download_button(
                    label="📥 Download as Excel",
                    data=to_excel(updated_df),
                    file_name="ner_results.xlsx",
                    #mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            with download_col2:
                st.download_button(
                    label="📥 Download as CSV",
                    data=to_csv(updated_df),
                    file_name="ner_results.csv",
                    #mime="text/csv",
                )

    if stop_button:
        st.session_state.stop_processing = True
        st.warning("Processing stopped by user.")

if __name__ == "__main__":
    main()
