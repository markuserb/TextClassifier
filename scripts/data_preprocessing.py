""" Daten werden geladen und vorverarbeitet (Tokenisierung, Entfernung von Stoppwörtern, Lemmatisierungn Normalisierung, Padding)"""
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk

# Lade NLTK-Ressourcen für Stoppwörter und Lemmatizer (falls noch nicht geschehen)
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(file_path):
    """Lädt die Daten aus einer CSV-Datei"""
    df = pd.read_csv(file_path)
    return df['text'], df['label']

def preprocess_texts(texts, max_words=10000, max_len=100):
    """Vorverarbeitet die Texte: Tokenisierung, Entfernung von Stoppwörtern, Lemmatisierung, Normalisierung und Padding"""
    
    # Entferne Stoppwörter
    texts = remove_stopwords(texts)

    # Lemmatisiere die Texte
    texts = lemmatize_texts(texts)

    # Textnormalisierung (Entfernen von Sonderzeichen, Zahlen, Umwandlung zu Kleinbuchstaben)
    texts = [normalize_text(text) for text in texts]
    
    # Tokenisierung
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Padding der Sequenzen
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    
    return padded_sequences, tokenizer

def normalize_text(text):
    """Normalisiert den Text: Entfernt Sonderzeichen, Zahlen und wandelt in Kleinbuchstaben um"""
    text = text.lower()  # Umwandlung in Kleinbuchstaben
    text = re.sub(r'[^a-z\s]', '', text)  # Entfernen aller nicht-alphabetischen Zeichen
    return text

def remove_stopwords(texts):
    """Entfernt Stoppwörter aus den Texten"""
    stop_words = set(stopwords.words('english'))  # Für Englisch
    return [' '.join([word for word in text.split() if word not in stop_words]) for text in texts]

def lemmatize_texts(texts):
    """Lemmatisiert die Texte"""
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in text.split()]) for text in texts]