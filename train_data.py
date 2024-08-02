from pipeline_traitement.cross_validation_score import cross_validation_score
from pipeline_traitement.set_up_pipeline import set_up_pipeline
import os
import numpy as np
import joblib


def train_data(epochs_data, labels):
    pipeline = set_up_pipeline()
    pipeline.fit(epochs_data, labels)
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, "models/pipeline.pkl")
    joblib.dump(pipeline, data_dir)
    scores = cross_validation_score(
        pipeline,
        epochs_data,
        labels,
    )

    print(scores)
    print(f"Classification accuracy: {np.mean(scores)} ")
