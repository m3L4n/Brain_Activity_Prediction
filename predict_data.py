import os
import joblib
import numpy as np

# from pipeline_traitement.cross_validation_score import cross_validation_score
from playback.playback import playback_reading
from sklearn.model_selection import cross_val_score


def predict_data(epochs_data, labels):
    """Predict with the pipeline downloaded with the train
    send to the playback and calculate the accuracy of the pipeline

    Parameters :
    epoch_data need to be preprocess
    epochs_data : data ndarray (n_epoch, n_channels, n_times)
    labels : the tag of each epoch  ndarray (n_epoch)
    """
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, "models/pipeline.pkl")
    pipeline = joblib.load(data_dir)
    playback_reading(epochs_data, pipeline, labels)
    scores = cross_val_score(
        pipeline,
        X=epochs_data,
        y=labels,
    )
    print(f"Mean cross_val_score {np.mean(scores)}")
