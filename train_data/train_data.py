from pipeline_traitement.set_up_pipeline import set_up_pipeline
import os
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
import joblib


def train_data(raw,  epochs_data, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        epochs_data, labels, test_size=0.2, random_state=42)
    pipeline = set_up_pipeline()
    pipeline.fit(X_train, y_train)
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, "../models/pipeline.pkl")
    joblib.dump(pipeline, data_dir)
    scores = cross_val_score(pipeline, epochs_data,
                             labels, cv=8, n_jobs=None)

    print(
        f"Classification accuracy: {np.mean(scores)} ")
