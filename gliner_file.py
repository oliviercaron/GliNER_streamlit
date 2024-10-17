from gliner import GLiNER
from typing import List

def run_ner(model, texts: List[str], labels_list: List[str], threshold=0.4):
    """
    Exécute la reconnaissance d'entités nommées (NER) sur une liste de textes.

    Paramètres:
    - model: modèle GLiNER chargé.
    - texts: liste de textes à analyser.
    - labels_list: liste des labels NER à détecter.
    - threshold: seuil de confiance pour les prédictions.

    Retourne:
    - ner_results: liste de dictionnaires contenant les entités détectées pour chaque texte.
    """
    ner_results = []
    for text in texts:
        try:
            # Prédire les entités pour le texte
            entities = model.predict_entities(text, labels_list, threshold=threshold)
            ner_results.append(entities)
        except Exception as e:
            ner_results.append([])
    return ner_results
