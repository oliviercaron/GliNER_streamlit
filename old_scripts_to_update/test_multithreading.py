import pandas as pd
import time
from datetime import timedelta
from gliner import GLiNER
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os

print("Current Working Directory:", os.getcwd())


def load_data(file_path):
    if file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)
    elif file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file type. Please use a CSV or XLSX file.")


def extract_entities(text, model, labels, threshold):
    # Extract named entities from the text
    entities = model.predict_entities(text, labels, threshold=threshold)
    entities_by_label = {label: [] for label in labels}
    for entity in entities:
        entities_by_label[entity["label"]].append(entity["text"])
    entities_by_label = {
        label: ", ".join(texts) for label, texts in entities_by_label.items()
    }
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

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            tqdm(
                executor.map(
                    lambda row: process_row(row, model, labels, threshold, column_name),
                    df.to_dict("records"),
                ),
                total=len(df),
            )
        )
        # results = list(tqdm(executor.map(lambda row: process_row(row, model, labels, threshold, column_name), df.to_dict('records'))))

    df_entities = pd.DataFrame(results)
    df_output = pd.concat([df, df_entities], axis=1)

    output_filename = "Dataframe_VF_datacovid_GliNER_part2_entities_output.csv"
    df_output.to_csv(output_filename, index=False)
    df_output.to_excel(
        "Dataframe_VF_datacovid_GliNER_part2_entities_output.xlsx",
        index=False,
        engine="xlsxwriter",
        engine_kwargs={"options": {"strings_to_urls": False}},
    )

    elapsed_time = time.time() - start_time
    print(df_output)
    print(f"Processing took {timedelta(seconds=elapsed_time)}.")


file_path = "test_data/Dataframe_VF_datacovid_GliNER_part2.csv"
labels = [
    "vaccine name",  # Specific names of COVID-19 vaccines
    "product name",  # Names of other pharmaceutical products mentioned
    "company name",  # Names of pharmaceutical companies
    "side effects",  # Side effects of vaccines
    "risks",  # Risks associated with vaccination
    "benefits",  # General benefits of vaccination
    "vaccine benefits",  # Specific benefits of vaccines
    "organization",  # Health and vaccination-related organizations
    "vaccine type",  # Types of vaccines (mRNA, viral vector, etc.)
    "regulatory status",  # Regulatory status of vaccines
    "clinical trial phase",  # Phases of clinical trials
    "vaccination campaign",  # Specific vaccination campaigns
    "adverse reaction",  # Serious adverse reactions discussed or reported
    "immunization rate",  # Vaccination coverage rates
    "public health policy",  # Public health policies related to vaccination
    "vaccine efficacy",  # Efficacy of vaccines in preventing cases, hospitalizations, or deaths
    "vaccine safety",  # Safety issues of vaccines, focusing on long-term monitoring
    "population segment",  # Specific population segments discussed in relation to vaccines
    "marketing strategy",  # Marketing strategies used by pharmaceutical companies
    "public opinion",  # Public opinion on vaccines, vaccine hesitancy, support, or controversy
    "government funding",  # Government funding or subsidies for vaccine research, development, or distribution
    "regulatory decision",  # Decisions made by regulatory agencies about vaccine approval, restrictions, or recommendations
    "economic impact",  # Economic impact of vaccination on health systems, businesses, and national economies
    "sickness name",
    #'vaccine accessibility',   # Availability and accessibility of vaccines in different regions or for different populations
    #'vaccine hesitancy',       # Specific discussions around reluctance or hesitation to get vaccinated
    #'policy debate',           # Political and legislative debates around vaccination policies
    #'health equity',           # Discussions on health equity concerning the fair distribution of vaccines and access for underserved populations
    #'vaccine diplomacy',       # Strategies used by governments to influence or assist other countries through vaccine distribution
    #'intellectual property'    # Discussions on intellectual property rights related to vaccines, including patents and licensing agreements
]
threshold = 0.3
column_name = "text"
model_name = "urchade/gliner_multi-v2.1"
process_entities(file_path, labels, threshold, column_name, model_name)
