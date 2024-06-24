import mne
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split

from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws
import pywt
from digest_data.detect_marks_bad_channels import detect_marks_bad_channels
from digest_data.montage_raw import montage_raw
from pipeline_traitement.set_up_pipeline import set_up_pipeline
from digest_data.filter_frequency import filter_edf


def wavelet_transform(data, wavelet='db4', level=4):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return np.concatenate(coeffs, axis=-1)


def test_all_data():
    tmin, tmax = -1.0, 4.0
    number_subject = 2
    task_dict = {1: [3, 7, 11],
                 2: [4, 8, 12],
                 3: [5, 9, 13],
                 4: [6, 10, 14],
                 5: [3, 7, 11, 4, 8, 12],
                 6: [5, 9, 12, 6, 10, 14]
                 }
    task_prediction = {
    }
    for i in range(1, number_subject):
        task_prediction[i] = {}
        for key, value in task_dict.items():
            task_prediction[i][key] = []

    for subject in range(1, number_subject):
        for task_id, runs in task_dict.items():
            raw_files = eegbci.load_data(subject, runs)
            raws = [mne.io.read_raw_edf(file, preload=True)
                    for file in raw_files]
            raw = concatenate_raws(raws)

            eegbci.standardize(raw)  # set channel names
            raw = montage_raw(raw)
            filter_edf(raw)
            detect_marks_bad_channels(raw)
            picks = pick_types(raw.info, meg=False, eeg=True,
                               stim=False, eog=False, exclude="bads")

            epochs = Epochs(
                raw,
                tmin=tmin,
                tmax=tmax,
                proj=True,
                picks=picks,
                baseline=None,
                preload=True,
            )
            epochs_train = epochs.copy().crop(tmin=1.0, tmax=4.0)
            labels = epochs.events[:, -1] - 1
            epochs_data_train = epochs_train.get_data(copy=False)

            epochs_data_train_wavelet = np.array(
                [wavelet_transform(epoch) for epoch in epochs_data_train])
            clf = set_up_pipeline()

            X_train, X_test, y_train, y_test = train_test_split(
                epochs_data_train_wavelet, labels, test_size=0.3, random_state=42)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            task_prediction[subject][task_id].append(
                accuracy_score(pred, y_test))
    task_res_dict = {}
    for i in range(1, 6):
        task_res_dict[i] = []

    for subject, result in task_prediction.items():

        print('result task', task_prediction, result)
        for exp_id, result_exp in result.items():
            print('result', result)
            print(
                f"Subject {subject} experiment {exp_id} accuracy {result_exp[0]}")
            task_res_dict[exp_id].append(result_exp[0])
    for key, value in task_res_dict.items():
        print(f"Experiment {key} accuracy{np.mean(value)}")
