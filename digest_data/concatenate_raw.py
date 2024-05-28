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


def concatenate_edf():
    subject = 4  # use data from subject 1

    runs = [14, 11, 9, 4, 1]
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(
        script_dir, '../physionet.org/files/eegmmidb/1.0.0/')
    files = eegbci.load_data(
        subject, runs, data_dir)

    raws = [read_raw_edf(f, preload=True) for f in files]
    # Combine all loaded runs
    raw_obj = concatenate_raws(raws)
    return raw_obj


def filter_edf(raw):
    raw.filter(8., 40., fir_design='firwin', skip_by_annotation='edge')
    raw.notch_filter(freqs=50.0, notch_widths=1.0)


def montage_raw(raw):

    montage_1 = mne.channels.make_standard_montage("standard_1005")

    ch_names = raw.ch_names

    ch_names = [x.replace(".", "") for x in ch_names]
    ch_names = [x.upper() for x in ch_names]

    montage_2 = [x.upper() for x in montage_1.ch_names]
    kept_channels = ch_names

    ind = [i for (i, channel) in enumerate(
        montage_2) if channel in kept_channels]
    montage_new = montage_1.copy()
    montage_new.ch_names = [montage_1.ch_names[x] for x in ind]

    kept_channel_info = [montage_1.dig[x+3] for x in ind]
    # Keep the first three rows as they are the fiducial points information
    montage_new.dig = montage_1.dig[0:3]+kept_channel_info

    montage_new.ch_names = [x.upper() for x in montage_new.ch_names]

    montage_names = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..',
                     'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..']
    montage_names_2 = [x.upper() for x in montage_names]
    montage_names_2 = [x.replace(".", "") for x in montage_names_2]

    montage_new_names = []
    for x in montage_new.ch_names:
        for i, s in enumerate(montage_names_2):
            if x == s:
                montage_new_names.append(i)
    montage_new.ch_names = [(montage_names[x]) for x in montage_new_names]
    raw = raw.set_montage(
        montage_new, match_case=False, match_alias=False)
    raw.set_eeg_reference('average', projection=True)
    return raw


# def remove_artifacts(raw):
#     ica = ICA(n_components=15, random_state=97)
#     ica.fit(raw)
#     # Automatic artifact detection based on kurtosis and variance
#     kurtosis_threshold = 5.0  # Adjust this threshold as needed
#     variance_threshold = 3.0  # Adjust this threshold as needed

#     sources = ica.get_sources(raw).get_data()
#     kurt = kurtosis(sources, axis=1)
#     variances = sources.var(axis=1)

#     artifact_indices = np.where((kurt > kurtosis_threshold) | (
#         variances > variance_threshold))[0]
#     ica.exclude = list(artifact_indices)
#     ica.plot_components()
#     plt.show()
#     ica.apply(raw)


def define_epochs(raw):
    # remove_artifacts(raw)
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
    epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)

    labels = epochs.events[:, -1]
    scores = []

    epochs_data_train = epochs_train.get_data(copy=False)
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    clf = Pipeline([("CSP", csp), ("LDA", lda)])
    # scores = cross_val_score(clf, epochs_data_train,
    #                          labels, cv=cv, n_jobs=None)

    # Printing the results
    # print(
    #     f"Classification accuracy: {np.mean(scores)} / ecart type: {np.std(scores)}",
    #     scores)
    epochs_data = np.asarray(epochs_data_train)
    labels = np.asarray(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        epochs_data, labels, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    correct = 0
    total = len(predictions)
    for i, (pred, truth) in enumerate(zip(predictions, y_test)):
        print(f"epoch {i:02d}: [{pred}] [{truth}] {pred == truth}")
        if pred == truth:
            correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")


def main():
    raw = concatenate_edf()

    variances = np.var(raw.get_data(), axis=1)
    threshold = np.mean(variances) + 3 * np.std(variances)
    bad_channels = [raw.ch_names[i]
                    for i in range(len(variances)) if variances[i] > threshold]
    raw.info['bads'] = bad_channels
    filter_edf(raw)

    raw = montage_raw(raw)
    define_epochs(raw)
    # raw.plot()
    # raw.plot_psd()
    plt.show()


if __name__ == "__main__":
    main()
