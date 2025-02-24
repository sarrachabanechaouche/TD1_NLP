# src/evaluate.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.corpus import stopwords

def evaluate(input_filename: str):
    
    #je fais une cross-validation (k-fold) et k=5 sur le CSV complet
    
    nltk.download('stopwords', quiet=True)
    french_stopwords = stopwords.words('french')
    
    data = pd.read_csv(input_filename)
    X = data['video_name']
    y = data['is_comic']
    
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(stop_words=french_stopwords)),
        ('classifier', LogisticRegression())
    ])
    
    scores = cross_val_score(pipeline, X, y, cv = 5, scoring='accuracy')
    print("Scores CV :", scores)
    print(f"Accuracy moyenne : {scores.mean():.3f} (+/- {scores.std():.3f})")
