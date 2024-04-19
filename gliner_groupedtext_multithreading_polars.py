import sys
import polars as pl
import time
from datetime import timedelta
from gliner import GLiNER
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os
import concurrent

def load_data(file_path, sample=False, n=None):
    if file_path.endswith('.xlsx'):
        df = pl.read_excel(file_path)
    elif file_path.endswith('.csv'):
        try:
            df = pl.read_csv(
                file_path,
                infer_schema_length=0,  # infer_schema_length=0 as polars uses string as default for columns type when reading csvs from https://stackoverflow.com/questions/71106690/polars-specify-dtypes-for-all-columns-at-once-in-read-csv
                #dtypes=dtypes, Otherwise we can specify dtypes for each column, which can be useful when we know the data types of the columns in advance, for instance utf8 for ids because it kept adding zeros to the ids when reading them as integers
                ignore_errors=True,
                low_memory=False,
                truncate_ragged_lines=True
            )
        except Exception as e:
            raise Exception(f"Failed to read CSV: {e}")
    else:
        raise ValueError("Unsupported file type. Please use a CSV or XLSX file.")
    
    if sample:
        if n is None:
            raise ValueError("Parameter 'n' must be specified if 'sample' is True.")
        df = df.sample(n=n, with_replacement=False)
    elif n is not None:
        df = df.head(n)
    
    return df

def extract_entities(text, model, labels, threshold):
    entities = model.predict_entities(text, labels=labels, threshold=threshold)
    entities_by_label = {label: [] for label in labels}
    for entity in entities:
        entities_by_label[entity["label"]].append(entity["text"])
    return entities_by_label

def process_row(text, model, labels, threshold):
    return extract_entities(text, model, labels, threshold)

def update_df_with_results(df, unique_texts, results, labels, column_name):
    # Ensure all label columns exist in the DataFrame
    for label in labels:
        if label not in df.columns:
            df = df.with_columns(pl.lit(None).alias(label))

    # Update DataFrame with results
    for i, result in enumerate(results):
        for label in labels:
            update_condition = df[column_name] == unique_texts[i]
            new_values = pl.lit(', '.join(result.get(label, [])))
            df = df.with_columns(
                pl.when(update_condition).then(new_values).otherwise(df[label]).alias(label)
            )
    return df

def process_entities(file_path, labels, threshold, column_name, model_name):
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return

    start_time = time.time()
    df = load_data(file_path, sample=False, n=100)
    if df.height == 0:
        print("No data to process.")
        return

    model = GLiNER.from_pretrained(model_name)
    model.eval()

    grouped_texts = df.group_by(column_name).agg(pl.len().alias('count'))
    unique_texts = grouped_texts.get_column(column_name).to_list()

    print("Processing unique texts...")
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_row, text, model, labels, threshold) for text in unique_texts]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            results.append(future.result())
    df = update_df_with_results(df, unique_texts, results, labels, column_name)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_csv_filename = f"{base_name}_entities_output.csv"
    output_xlsx_filename = f"{base_name}_entities_output.xlsx"

    df.write_csv(output_csv_filename)
    df.write_excel(output_xlsx_filename, autofit=True)

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