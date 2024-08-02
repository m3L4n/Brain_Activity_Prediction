from pipeline_traitement.cross_validation_score import cross_validation_score
from pipeline_traitement.set_up_pipeline import set_up_pipeline
import os
import numpy as np
import joblib


def train_data(epochs_data, labels):
    """
    Train script that receives preprocessing data and fit (train)
    the Pipeline(csp and classifier(lda)) this  process will save weight of
    our algorithmn csp and lda and we will save this model and evaluate the
    accuracy of this model

    Parameters :
    epoch_data need to be preprocess
    epochs_data : data ndarray (n_epoch, n_channels, n_times)
    labels : the tag of each epoch  ndarray (n_epoch)
    """
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
