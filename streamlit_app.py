import streamlit as st
import polars as pl
from io import BytesIO
from gliner import GLiNER
from gliner_file import run_ner

st.set_page_config(page_title="GliNER", page_icon="🧊", layout="wide", initial_sidebar_state="expanded")

# Load GLiNER model if not already loaded
if 'gliner_model' not in st.session_state:
    with st.spinner('Loading GLiNER model... Please wait.'):
        st.session_state.gliner_model = GLiNER.from_pretrained("urchade/gliner_largev2")
        st.session_state.gliner_model.eval()

# Function to load data from an Excel file
@st.cache_data
def load_data(file):
    return pl.read_excel(file)

# Function to perform NER and update the UI
def perform_ner(filtered_df, selected_column, labels_list):

    series_dict = {label: pl.Series(name=label, values=[""] * filtered_df.height) for label in labels_list}
    progress_bar = st.progress(0)
    progress_text = st.empty()

    for index, row in enumerate(filtered_df.to_pandas().itertuples(), 1):
        if st.session_state.stop_processing:
            progress_text.text("Process stopped by the user.")
            break

        text_to_analyze = getattr(row, selected_column)
        ner_results = run_ner(st.session_state.gliner_model, text_to_analyze, labels_list)

        for label, texts in ner_results.items():
            concatenated_texts = ', '.join(texts)
            series_dict[label][row.Index] = concatenated_texts

        progress = index / filtered_df.height
        progress_bar.progress(progress)
        progress_text.text(f"Progress: {index}/{filtered_df.height} - {progress * 100:.0f}%")

    progress_text.text("Processing complete!" if not st.session_state.stop_processing else "")
    return filtered_df.with_columns(list(series_dict.values()))

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
    
    df = load_data(uploaded_file)
    selected_column = st.selectbox("Select the column for NER:", df.columns, index=0)
    filter_text = st.text_input("Filter column by input text", "")
    ner_labels = st.text_input("Enter all your different labels, separated by a comma", "")

    filtered_df = df.filter(pl.col(selected_column).str.contains(f"(?i).*{filter_text}.*")) if filter_text else df
    st.dataframe(filtered_df)

    if st.button("Start NER"):
        if not ner_labels:
            st.warning("Please enter some labels for NER.")
        else:
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
