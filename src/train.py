
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.corpus import stopwords

def train(input_filename: str, model_dump_filename: str):

    # ici on va 1-Lire le CSV (chemin = input_filename)  
    # 2_Séparerr le X et le Y   
    # 3- Spliter en train/test 80/20 
    # 4-Entraîner un pipeline (CountVectorizer + LogisticRegression) 
    # 5-Évaluer sur le test (accuracy, classification report)  
    # 6- Sauvegarde le pipeline dans model_dump_filenamee
   
    
    nltk.download('stopwords', quiet=True)
    french_stopwords = stopwords.words('french')
   
    data = pd.read_csv(input_filename)
    X = data['video_name']
    y = data['is_comic']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    
    # Pipeline : CountVectorizer(stop_words=french_stopwords) + LogisticRegression
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(
            stop_words=french_stopwords,  # on passe la liste des stopwords français
            min_df=1,
            max_df=0.9
        )),
        ('classifier', LogisticRegression())
    ])
    
    # Entraînement
    pipeline.fit(X_train, y_train)
    
    # Évaluation sur le split 20%
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy (split 20%) : {acc:.3f}")
    print("Classification report :")
    print(classification_report(y_test, y_pred))
    
    # Sauvegarde du pipeline
    joblib.dump(pipeline, model_dump_filename)
    print(f"Modèle sauvegardé dans : {model_dump_filename}")
