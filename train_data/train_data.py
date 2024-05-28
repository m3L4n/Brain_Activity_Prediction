from pipeline_traitement.set_up_pipeline import set_up_pipeline
import os
import numpy as np
import pandas as pd
import mne
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split
import joblib
from mne import Epochs, pick_types, find_events, pick_types, set_eeg_reference
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from mne import viz
from mne.preprocessing import ICA, create_eog_epochs
from scipy.stats import kurtosis

import matplotlib.pyplot as plt


def train_data(raw,  epochs_data, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        epochs_data, labels, test_size=0.2, random_state=42)
    pipeline = set_up_pipeline()
    pipeline.fit(X_train, y_train)
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, "../models/pipeline.pkl")
    joblib.dump(pipeline, data_dir)
    y_pred = pipeline.predict(X_test)
    scores = cross_val_score(pipeline, epochs_data,
                             labels, cv=8, n_jobs=None)
    print(
        f"Classification accuracy: {np.mean(scores)} / ecart type: {np.std(scores)}",
        scores, y_pred)
