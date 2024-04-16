import pandas as pd
import time
from datetime import timedelta
from gliner import GLiNER
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os

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

def process_entities(file_path, labels, threshold, column_name, model_name, output_base):
    start_time = time.time()
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to load data from {file_path}: {e}")
        return
    
    try:
        model = GLiNER.from_pretrained(model_name)
    except Exception as e:
        print(f"Failed to load the model {model_name}: {e}")
        return
    model.eval()

    try:
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(tqdm(executor.map(lambda row: process_row(row, model, labels, threshold, column_name), df.to_dict('records')), total=len(df)))
        df_entities = pd.DataFrame(results)
        df_output = pd.concat([df, df_entities], axis=1)
    except Exception as e:
        print(f"Error during entity processing: {e}")
        return

    output_csv = f"{output_base}_entities_output.csv"
    output_xlsx = f"{output_base}_entities_output.xlsx"
    df_output.to_csv(output_csv, index=False)
    df_output.to_excel(output_xlsx, index=False)

    elapsed_time = time.time() - start_time
    print(f"Processing of {output_csv} took {timedelta(seconds=elapsed_time)}.")


labels = [
    'vaccine name',            # Specific names of COVID-19 vaccines
    'product name',            # Names of other pharmaceutical products mentioned
    'company name',            # Names of pharmaceutical companies
    'side effects',            # Side effects of vaccines
    'risks',                   # Risks associated with vaccination
    'benefits',                # General benefits of vaccination
    'vaccine benefits',        # Specific benefits of vaccines
    'organization',            # Health and vaccination-related organizations
    'vaccine type',            # Types of vaccines (mRNA, viral vector, etc.)
    'regulatory status',       # Regulatory status of vaccines
    'clinical trial phase',    # Phases of clinical trials
    'vaccination campaign',    # Specific vaccination campaigns
    'adverse reaction',        # Serious adverse reactions discussed or reported
    'immunization rate',       # Vaccination coverage rates
    'public health policy',    # Public health policies related to vaccination
    'vaccine efficacy',        # Efficacy of vaccines in preventing cases, hospitalizations, or deaths
    'vaccine safety',          # Safety issues of vaccines, focusing on long-term monitoring
    'population segment',      # Specific population segments discussed in relation to vaccines
    'marketing strategy',      # Marketing strategies used by pharmaceutical companies
    'public opinion',          # Public opinion on vaccines, vaccine hesitancy, support, or controversy
    'government funding',      # Government funding or subsidies for vaccine research, development, or distribution
    'regulatory decision',     # Decisions made by regulatory agencies about vaccine approval, restrictions, or recommendations
    'economic impact',         # Economic impact of vaccination on health systems, businesses, and national economies
    'sickness name'
    #'vaccine accessibility',   # Availability and accessibility of vaccines in different regions or for different populations
    #'vaccine hesitancy',       # Specific discussions around reluctance or hesitation to get vaccinated
    #'policy debate',           # Political and legislative debates around vaccination policies
    #'health equity',           # Discussions on health equity concerning the fair distribution of vaccines and access for underserved populations
    #'vaccine diplomacy',       # Strategies used by governments to influence or assist other countries through vaccine distribution
    #'intellectual property'    # Discussions on intellectual property rights related to vaccines, including patents and licensing agreements
]

threshold = 0.3
column_name = 'text'
model_name = "urchade/gliner_multi-v2.1"
base_dir = "C:/Users/Olivier/Documents/GitHub/GliNER_streamlit/test_data"  # Adjust path for files
base_output = "C:/Users/Olivier/Documents/GitHub/GliNER_streamlit/output_data"  # Adjust path for output files
parts = ['part1', 'part2', 'part3', 'part4']

for part in parts:
    file_path = os.path.join(base_dir, f"Dataframe_VF_datacovid_GliNER_{part}.csv")
    output_base = os.path.join(base_output, f"Dataframe_VF_datacovid_GliNER_{part}")
    process_entities(file_path, labels, threshold, column_name, model_name, output_base)