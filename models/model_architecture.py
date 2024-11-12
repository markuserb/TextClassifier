""" Modellarchitektur mit zwei LSTM Schichten, Dropoutschichten, sowie Adam Optimizer und binary_crossentropy als Verlustfunktion"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2

def build_model(input_dim, output_dim, input_length):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length),
        LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        LSTM(64, kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))  # Für binäre Klassifikation
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model