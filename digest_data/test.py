from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
import os
from sklearn.decomposition import PCA
import mne
from mne.channels import make_standard_montage
from mne.io import read_raw_edf, concatenate_raws
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from mne import Epochs, pick_types
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline


def get_directory_data():
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(
        script_dir, '../physionet.org/files/eegmmidb/1.0.0/')
    return data_dir


def get_concatenate_edf(index_max=2):

    directory = get_directory_data()
    elems = os.listdir(directory)
    elems.sort()
    index = 0
    raw_list = []
    for elem in elems:
        if index > index_max:
            break
        if os.path.isdir(directory + elem):
            items = os.listdir(directory + elem)
            items.sort()
            for edf in items:
                if edf.endswith('.edf'):
                    raw = mne.io.read_raw_edf(
                        directory + elem + "/" + edf, preload=True)
                    if raw.info['sfreq'] != 160:
                        raw.resample(160)
                    raw_list.append(raw)
            index += 1
    return mne.concatenate_raws(raw_list)


def filter_frequency(raw):

    raw.filter(8, 40, fir_design='firwin')
    # raw.plot_psd()
    # plt.show()
    # fig = raw.compute_psd(tmax=np.inf, fmax=80).plot(
    #     average=True, amplitude=False, picks="data", exclude="bads"
    # )
    plt.show()


def show_raw_plot(raw):
    raw.plot_psd()  # to see the psd  with the different frequency
    raw.plot()
    plt.show()


def main():
    raw = get_concatenate_edf()
    filter_frequency(raw)
    tmin, tmax = -1.0, 4.0
    picks = pick_types(raw.info, meg=False, eeg=True,
                       stim=False, eog=False, exclude="bads")
    events, event_id = mne.events_from_annotations(raw)

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
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
    epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)

    labels = epochs.events[:, -1] - 2
    scores = []

    epochs_data = epochs.get_data(copy=False)
    epochs_data_train = epochs_train.get_data(copy=False)
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([("CSP", csp), ("LDA", lda)])
    scores = cross_val_score(clf, epochs_data_train,
                             labels, cv=cv, n_jobs=None)

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1.0 - class_balance)
    print(
        f"Classification accuracy: {np.mean(scores)} / Chance level: {class_balance}")

    # plot CSP patterns estimated on full data for visualization
    csp.fit_transform(epochs_data, labels)

    csp.plot_patterns(epochs.info, ch_type="eeg",
                      units="Patterns (AU)", size=1.5)


# plt.show()


if __name__ == "__main__":
    main()
