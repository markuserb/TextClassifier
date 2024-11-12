""" Hauptprogramm """

import os, sys
# Füge das übergeordnete Verzeichnis zum sys.path hinzu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.import_data import import_dataset
from data_preprocessing import load_data, preprocess_texts, remove_stopwords, lemmatize_texts
from models.model_architecture import build_model
from train_model import train_and_save_model
from evaluate_model import evaluate_model

def main():
    # Lade den IMDb-Datensatz
    import_dataset()

    # Lade die Trainings- und Testdaten
    train_texts, train_labels = load_data('data/train.csv')
    test_texts, test_labels = load_data('data/test.csv')

    # Setze maximale Anzahl an Wörtern und maximale Sequenzlänge für Tokenisierung und Padding
    max_words = 10000
    max_len = 100

    # Tokenisierung der Texte
    train_sequences, train_tokenizer = preprocess_texts(train_texts, max_words, max_len)
    test_sequences, _ = preprocess_texts(test_texts, max_words, max_len)

    # Trainiere und speichere das Modell
    train_and_save_model(train_sequences, train_labels, max_words=max_words, max_len=max_len)

    # Baue das Modell erneut, um es für die Evaluierung zu laden
    model = build_model(input_dim=max_words, output_dim=100, input_length=max_len)

    # Evaluiere das Modell
    evaluate_model(model, test_sequences, test_labels)

if __name__ == "__main__":
    main()