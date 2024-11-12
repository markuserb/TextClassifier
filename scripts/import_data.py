""" Daten werden geladen """

import os
from datasets import load_dataset

def import_dataset():
    # Lade den Datensatz
    ds = load_dataset("jahjinx/IMDb_movie_reviews")
    

    # Stelle sicher, dass der Ordner 'data' existiert
    os.makedirs('data', exist_ok=True)

    # Speichere die Trainings- und Testdaten als CSV-Dateien im Ordner 'data'
    train_data = ds['train']
    test_data = ds['test']

    # Speichere als CSV-Dateien
    train_data.to_pandas().to_csv('data/train.csv', index=False)
    test_data.to_pandas().to_csv('data/test.csv', index=False)

    print("Daten wurden im 'data'-Ordner gespeichert.")