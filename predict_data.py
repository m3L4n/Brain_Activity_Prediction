import os
import joblib
import numpy as np
from playback.playback import playback_reading
from sklearn.model_selection import cross_val_score


def predict_data(epochs_data, labels):
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, "models/pipeline.pkl")
    pipeline = joblib.load(data_dir)
    playback_reading(epochs_data, pipeline, labels)
    scores = cross_val_score(pipeline, epochs_data, labels)
    print(f"Accuracy {np.mean(scores)}")
