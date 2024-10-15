from gliner import GLiNER

def run_ner(model, text, labels_list, threshold=0.4):
    # Utiliser le modèle déjà chargé pour prédire les entités
    entities = model.predict_entities(text, labels_list, threshold=threshold)

    # Initialiser le dictionnaire pour stocker les résultats
    ner_results = {label: [] for label in labels_list}

    # Itérer sur les entités reconnues et les stocker dans le dictionnaire
    for entity in entities:
        if entity["label"] in ner_results:
            # Ajouter le texte de l'entité à la liste correspondante pour le label
            ner_results[entity["label"]].append(entity["text"])

    return ner_results
