# src/predict.py

import pandas as pd
import joblib

def predict(input_filename: str, model_dump_filename: str, output_filename: str):
    
    #- Charge le pipeline (model + vectorizer).
    #- Lit input_filename (CSV) qui doit avoir 'video_name'.
    #- Fait la prédiction.
    
    # Charger le pipeline
    pipeline = joblib.load(model_dump_filename)
    
    # Lire le CSV
    data = pd.read_csv(input_filename)
    if 'video_name' not in data.columns:
        raise ValueError("Le CSV d'entrée doit contenir la colonne 'video_name'.")
    
    X = data['video_name']
    
    # Prédiction
    y_pred = pipeline.predict(X)
    
    # Sauvegarde
    data['prediction_is_comic'] = y_pred
    data.to_csv(output_filename, index=False)
    print(f"Prédictions écrites dans {output_filename}")
