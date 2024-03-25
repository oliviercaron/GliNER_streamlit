from gliner import GLiNER

def run_ner(model, text, labels_list, threshold=0.4):

    entities = model.predict_entities(text, labels_list, threshold=threshold)

    # Chargement du modèle GLiNER
    model = GLiNER.from_pretrained("urchade/gliner_largev2")
    model.eval()  # Mettre le modèle en mode évaluation
    
    # Initialisation du dictionnaire pour stocker les résultats
    ner_results = {label: [] for label in labels_list}

    # Parcourir les entités reconnues et les stocker dans le dictionnaire
    for entity in entities:
        if entity['label'] in ner_results:
            # Ajoute le texte de l'entité à la liste correspondante au label
            ner_results[entity['label']].append(entity['text'])

    return ner_results
