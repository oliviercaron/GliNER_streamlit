import sys
import pandas as pd
import time
from datetime import timedelta
from gliner import GLiNER
from tqdm import tqdm
import concurrent
from concurrent.futures import ThreadPoolExecutor
import os

def load_data(file_path, sample=False, n=None):
    # Determine the file type and read the data
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig', on_bad_lines='skip', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='iso-8859-1')
    else:
        raise ValueError("Unsupported file type. Please use a CSV or XLSX file.")
    
    if sample:
        if n is None:
            raise ValueError("Parameter 'n' must be specified if 'sample' is True.")
        df = df.sample(n=n)
    elif n is not None:
        df = df.head(n)
    
    return df

def extract_entities(text, model, labels, threshold):
    # Extract named entities from the text
    entities = model.predict_entities(text, labels, threshold=threshold)
    entities_by_label = {label: [] for label in labels}
    for entity in entities:
        entities_by_label[entity["label"]].append(entity["text"])
    return {label: ', '.join(texts) if texts else '' for label, texts in entities_by_label.items()}

def process_row(text, model, labels, threshold):
    return extract_entities(text, model, labels, threshold)

def process_entities(file_path, labels, threshold, column_name, model_name):
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return

    start_time = time.time()
    df = load_data(file_path, sample=False, n=100)
    if df.empty:
        print("No data to process.")
        return

    model = GLiNER.from_pretrained(model_name)
    model.eval()

    grouped_texts = df.groupby(column_name).size().reset_index(name='count')
    unique_texts = grouped_texts[column_name].tolist()

    with ThreadPoolExecutor(max_workers=4) as executor:
        unique_results = list(tqdm(executor.map(lambda text: process_row(text, model, labels, threshold), unique_texts), total=len(unique_texts)))

    results_dict = dict(zip(unique_texts, unique_results))
    for label in labels:
        df[label] = df[column_name].map(lambda x: results_dict[x].get(label, ''))

    output_dir = "output_data"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_csv_filename = os.path.join(output_dir, f"{base_name}_entities_output.csv")
    output_xlsx_filename = os.path.join(output_dir, f"{base_name}_entities_output.xlsx")

    df.to_csv(output_csv_filename, index=False)
    df.to_excel(output_xlsx_filename, index=False, engine='xlsxwriter')

    elapsed_time = time.time() - start_time
    print(f"Processing took {timedelta(seconds=elapsed_time)}.")

if __name__ == "__main__":
    print("Current Working Directory:", os.getcwd())
    if len(sys.argv) != 2:
        print("Usage: python <script_name.py> <file_path>")
    else:
        file_path = sys.argv[1]
        labels = [
            'vaccine name', 'product name', 'company name', 'side effects',
            'risks', 'benefits', 'vaccine benefits', 'organization', 'vaccine type',
            'regulatory status', 'clinical trial phase', 'vaccination campaign',
            'adverse reaction', 'immunization rate', 'public health policy',
            'vaccine efficacy', 'vaccine safety', 'population segment', 'marketing strategy',
            'public opinion', 'government funding', 'regulatory decision',
            'economic impact', 'sickness name'
        ]
        threshold = 0.3
        column_name = 'text'
        model_name = "urchade/gliner_multi-v2.1"
        process_entities(file_path, labels, threshold, column_name, model_name)