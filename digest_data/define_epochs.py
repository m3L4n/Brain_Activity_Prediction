import os
import numpy as np
import pandas as pd
import mne
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split

from mne import Epochs, pick_types, find_events, pick_types, set_eeg_reference
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from mne import viz
from mne.preprocessing import ICA, create_eog_epochs
from scipy.stats import kurtosis

import matplotlib.pyplot as plt


def define_epochs(raw):
    events, event_id = mne.events_from_annotations(raw)
    picks = pick_types(raw.info, meg=False, eeg=True,
                       stim=False, eog=False, exclude="bads")
    tmin, tmax = -1.0, 4.0
    epochs = Epochs(
        raw,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )
    print(f"Number of epochs: {len(epochs)}")
    epochs_train = epochs.copy()

    labels = epochs.events[:, -1]

    epochs_data_train = epochs_train.get_data(copy=False)
    return epochs_data_train, labels
