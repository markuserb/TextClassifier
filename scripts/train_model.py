import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models.model_architecture import build_model
from models.model_save_load import save_model
from tensorflow.keras.callbacks import ModelCheckpoint

def train_and_save_model(padded_sequences, labels, max_words=10000, max_len=50):
    """
    Diese Funktion führt alle Schritte des Trainings aus: 
    - Vorverarbeitung der Texte
    - Modell erstellen
    - Modell trainieren und speichern
    """

    # Modell erstellen
    model = build_model(input_dim=max_words, output_dim=100, input_length=max_len)

    # Modell Checkpoint: Speichert das beste Modell basierend auf der Validierungsgenauigkeit
    checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

    # Modell trainieren
    model.fit(padded_sequences, labels, epochs=5, batch_size=32, validation_split=0.2, callbacks=[checkpoint])

    # Pfad zum Modell im übergeordneten Verzeichnis
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'final_model.keras')

    # Modell speichern
    save_model(model, model_path)

# Beispieltext und Labels für das Testen der Funktion
if __name__ == "__main__":
    texts = ["Das ist ein Beispieltext.", "Noch ein Text für das Modell."]
    labels = [0, 1]

    # Modell trainieren und speichern
    train_and_save_model(texts, labels)