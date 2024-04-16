import sys
import pandas as pd
import time
from datetime import timedelta
from gliner import GLiNER
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os

print("Current Working Directory:", os.getcwd())

def load_data(file_path):
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file type. Please use a CSV or XLSX file.")

def extract_entities(text, model, labels, threshold):
    # Extract named entities from the text
    entities = model.predict_entities(text, labels, threshold=threshold)
    entities_by_label = {label: [] for label in labels}
    for entity in entities:
        entities_by_label[entity["label"]].append(entity["text"])
    entities_by_label = {label: ', '.join(texts) for label, texts in entities_by_label.items()}
    return entities_by_label

def process_row(row, model, labels, threshold, column_name):
    # Process a single row of data
    return extract_entities(row[column_name], model, labels, threshold)

def process_entities(file_path, labels, threshold, column_name, model_name):
    # Check if the file exists before proceeding
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return

    start_time = time.time()
    df = load_data(file_path)
    model = GLiNER.from_pretrained(model_name)
    model.eval()

    print("Processing data...")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(tqdm(executor.map(lambda row: process_row(row, model, labels, threshold, column_name), df.to_dict('records')), total=len(df)))

    df_entities = pd.DataFrame(results)
    df_output = pd.concat([df, df_entities], axis=1)

    output_filename = "Dataframe_VF_datacovid_GliNER_part2_entities_output.csv"
    df_output.to_csv(output_filename, index=False)
    df_output.to_excel("Dataframe_VF_datacovid_GliNER_part2_entities_output.xlsx", index=False, engine='xlsxwriter', engine_kwargs={'options': {'strings_to_urls': False}})

    elapsed_time = time.time() - start_time
    print(df_output)
    print(f"Processing took {timedelta(seconds=elapsed_time)}.")

if __name__ == "__main__":
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
