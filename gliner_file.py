from gliner import GLiNER


def run_ner(model, text, labels_list, threshold=0.4):

    entities = model.predict_entities(text, labels_list, threshold=threshold)

    # Loading the GLiNER model
    model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    model.eval()  # Put the model in evaluation mode

    # Initializing the dictionary to store the results
    ner_results = {label: [] for label in labels_list}

    # Iterating over the recognized entities and storing them in the dictionary
    for entity in entities:
        if entity["label"] in ner_results:
            # Adds the entity's text to the corresponding list for the label
            ner_results[entity["label"]].append(entity["text"])

    return ner_results
