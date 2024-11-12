""" Evaluierung des Models """

import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def evaluate_model(model, test_sequences, test_labels):

    # Modell evaluieren
    loss, accuracy = model.evaluate(test_sequences, test_labels)
    print(f"Loss: {loss}, Accuracy: {accuracy}")