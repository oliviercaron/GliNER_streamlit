import os 
import csv 
import streamlit as st  # For creating the interactive web app
import polars as pl  # Same as Pandas but faster and I wanted to try
from io import BytesIO, StringIO  # For handling file data in memory
from gliner import GLiNER  # Importing GLiNER model for NER
from gliner_file import run_ner  # Function to run NER on given text
import time 
import torch  # For checking GPU availability and using GPU with cuda
import platform  # For getting CPU information

# Configure the Streamlit page
st.set_page_config(
    page_title="GliNER", page_icon="🔥", layout="wide", initial_sidebar_state="expanded"
)

# Function to load data from an uploaded CSV or Excel file
@st.cache_data
def load_data(file):
    _, file_ext = os.path.splitext(file.name)
    
    # Check the file extension and read accordingly
    if file_ext.lower() in [".xls", ".xlsx"]:
        return pl.read_excel(file)
    elif file_ext.lower() == ".csv":
        file.seek(0)  # Reset file pointer to start
        try:
            # Attempt UTF-8 decoding
            sample = file.read(4096).decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            # Fall back to Latin-1 encoding if UTF-8 fails
            encoding = "latin1"
            file.seek(0)
            sample = file.read(4096).decode(encoding)

        file.seek(0)
        # Detect CSV delimiter
        dialect = csv.Sniffer().sniff(sample)

        file.seek(0)
        # Load CSV data with Polars using the detected delimiter
        if encoding != "utf-8":
            file_content = file.read().decode(encoding)
            file = StringIO(file_content)
        else:
            file_content = file.read().decode("utf-8")
            file = StringIO(file_content)

        return pl.read_csv(
            file,
            separator=dialect.delimiter,
            truncate_ragged_lines=True,
            ignore_errors=True,
        )
    else:
        raise ValueError("Uploaded file must be a CSV or Excel file.")

# Function to load the GLiNER model only once
def load_model():
    if "gliner_model" not in st.session_state:
        # Check for GPU availability
        st.session_state.gpu_available = torch.cuda.is_available()

        with st.spinner("Loading GLiNER model... Please wait."):
            # Load model to GPU if available, else to CPU
            if st.session_state.gpu_available:
                device = torch.device("cuda")
                st.session_state.gliner_model = GLiNER.from_pretrained(
                    "urchade/gliner_multi-v2.1"
                ).to(device)
            else:
                device = torch.device("cpu")
                st.session_state.gliner_model = GLiNER.from_pretrained(
                    "urchade/gliner_multi-v2.1"
                ).to(device)
            st.session_state.gliner_model.eval()

        # Display message about where the model is loaded
        if st.session_state.gpu_available:
            device_name = torch.cuda.get_device_name(0)
            st.success(f"GPU detected: {device_name}. Model loaded on GPU.")
        else:
            cpu_name = platform.processor()
            st.warning(f"GPU not detected. Using CPU: {cpu_name}")

# Function to perform NER and update the UI
def perform_ner(filtered_df, selected_column, labels_list, threshold):
    # Initialize dictionary to store results per label
    ner_results_dict = {label: [] for label in labels_list}

    # UI elements for progress tracking
    progress_bar = st.progress(0)
    progress_text = st.empty()

    start_time = time.time()
    total_rows = filtered_df.shape[0]

    # Loop over each row in the DataFrame to apply NER
    for index, row in enumerate(filtered_df.to_pandas().itertuples(), 1):
        if st.session_state.stop_processing:
            # Stop if requested
            progress_text.text("Processing stopped by the user.")
            break

        text_to_analyze = getattr(row, selected_column)
        # Run NER on the text
        ner_results = run_ner(
            st.session_state.gliner_model, text_to_analyze, labels_list, threshold=threshold
        )

        for label in labels_list:
            # Join results per label with commas if multiple results
            texts = ner_results.get(label, [])
            concatenated_texts = ", ".join(texts) if texts else ""
            ner_results_dict[label].append(concatenated_texts)

        # Update progress bar and text
        progress = index / total_rows
        progress_bar.progress(progress)

        iteration_time = time.time() - start_time
        progress_text.text(
            f"Progress: {index}/{total_rows} - {progress * 100:.0f}% (Elapsed: {iteration_time:.2f}s)"
        )

    end_time = time.time()
    if not st.session_state.stop_processing:
        progress_text.text(f"Processing complete! Total time: {end_time - start_time:.2f}s")
    else:
        progress_text.text("Processing was stopped.")

    # Add results to the DataFrame
    for label, texts in ner_results_dict.items():
        if len(texts) != total_rows:
            st.error(f"Length mismatch for label '{label}' ({len(texts)}) and DataFrame rows ({total_rows}).")
        else:
            filtered_df = filtered_df.with_columns(pl.Series(name=label, values=texts))

    return filtered_df

# Main function to run the Streamlit app
def main():
    st.title("Online NER with GliNER")
    st.markdown("Prototype v0.1")

    # Initialize session state variables
    if "stop_processing" not in st.session_state:
        st.session_state.stop_processing = False
    if "threshold" not in st.session_state:
        st.session_state.threshold = 0.4  # Default threshold value

    # Load the model once at the start
    load_model()

    # File upload sidebar
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is None:
        st.warning("Please upload a file.")
        return

    # Load data from the file
    try:
        df = load_data(uploaded_file)
    except ValueError as e:
        st.error(str(e))
        return

    # UI components for user input
    selected_column = st.selectbox("Select column for NER:", df.columns, index=0)
    filter_text = st.text_input("Filter column by text", "")
    ner_labels = st.text_input("Enter your labels, separated by commas", "")

    # Directly bind the slider to st.session_state.threshold
    st.slider("Set confidence threshold", 0.0, 1.0, st.session_state.threshold, 0.01, key="threshold")

    # Filter DataFrame based on user input
    filtered_df = (
        df.filter(pl.col(selected_column).str.contains(f"(?i).*{filter_text}.*"))
        if filter_text
        else df
    )
    st.dataframe(filtered_df, use_container_width=True)

    # Buttons for starting and stopping NER
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Start NER")
    with col2:
        stop_button = st.button("Stop")

    if start_button:
        # Reset stop_processing before starting new NER process
        st.session_state.stop_processing = False

        if not ner_labels:
            st.warning("Please enter labels for NER.")
        else:
            labels_list = [label.strip() for label in ner_labels.split(",") if label.strip()]
            updated_df = perform_ner(filtered_df, selected_column, labels_list, st.session_state.threshold)
            st.dataframe(updated_df, use_container_width=True)

            # Helper function to download results as Excel
            def to_excel(df):
                output = BytesIO()  # Create an in-memory bytes buffer
                df.to_pandas().to_excel(output, index=False, engine="openpyxl")  # Write Excel data to this buffer
                return output.getvalue()  # Retrieve the bytes content of the buffer

            # Helper function to download results as CSV
            def to_csv(df):
                return df.to_pandas().to_csv(index=False).encode("utf-8")

            # Convert DataFrame to download formats
            df_excel = to_excel(updated_df)
            df_csv = to_csv(updated_df)

            # Display download buttons for Excel and CSV side by side
            download_col1, download_col2 = st.columns(2)
            with download_col1:
                st.download_button(
                    label="📥 Download as Excel",
                    data=df_excel,
                    file_name="ner_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            with download_col2:
                st.download_button(
                    label="📥 Download as CSV",
                    data=df_csv,
                    file_name="ner_results.csv",
                    mime="text/csv",
                )

    # Stop processing if the stop button is clicked
    if stop_button:
        st.session_state.stop_processing = True

if __name__ == "__main__":
    main()
