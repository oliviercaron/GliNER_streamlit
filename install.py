import pandas as pd
from gliner import GLiNER
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Initialize GLiNER model
model = GLiNER.from_pretrained("urchade/gliner_largev2")
model.eval()

# Load the DataFrame from the Excel file
df_tweets = pd.read_excel("personal_fake_tweets_en.xlsx")

# Define labels for entity extraction
labels = ["person", "book", "location", "date", "emotion", "company_name", "job_title", "product_name", "event", "organization", "other"]

# Function to predict entities and return them in a structured format
def extract_entities(text):
    entities = model.predict_entities(text, labels, threshold=0.4)
    # Organize entities by label
    entities_by_label = {label: [] for label in labels}
    for entity in entities:
        entities_by_label[entity["label"]].append(entity["text"])
    # Convert lists to comma-separated strings
    entities_by_label = {label: ', '.join(texts) for label, texts in entities_by_label.items()}
    return entities_by_label

# Function to apply extraction to a single row
def process_row(row):
    return extract_entities(row['text'])

# Apply the function to each text in the DataFrame with ThreadPoolExecutor
with tqdm(total=len(df_tweets)) as pbar:
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_row, df_tweets.to_dict('records')))
        pbar.update()

# Convert results to DataFrame
df_entities = pd.DataFrame(results)

# Concatenate the original text column with the extracted entities DataFrame
df_output = pd.concat([df_tweets, df_entities], axis=1)

# Save the DataFrame to an Excel file
df_output.to_excel("entities_output_fastest2.xlsx", index=False)

print(df_output)
