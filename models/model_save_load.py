from tensorflow.keras.models import load_model

def save_model(model, file_path):
    model.save(file_path)

def load_model_from_file(file_path):
    return load_model(file_path)