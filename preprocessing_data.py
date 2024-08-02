import os
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne import Epochs, pick_types
from mne import (
    events_from_annotations,
)
from mne.channels import make_standard_montage


def concatenate_edf(subject_number: int, runs_experiment: list):
    """Load all the edf file corresponding at the nbr of subject and the runs
    experiments. Its make the montage of concatenated files
    Return the object Raw of all experiments

    Parameters:
    subject_number : int between 1 to 109
    runs)experiment : list of int between 3 to 14

    """
    try:
        assert isinstance(subject_number, int), "Error subject number has to be int"
        assert isinstance(runs_experiment, list), "Error runs_experiment has to be list"
        assert all(
            [int(x) for x in runs_experiment]
        ), "Error runs_experiment has to be a list of int"
        assert all(
            [x >= 3 and x <= 14 for x in runs_experiment]
        ), "Error runs_experiment need to be in 3 and 14"
        assert (
            subject_number >= 1 and subject_number <= 109
        ), "Error subject_number need to be in 1 and 109"
        subject = subject_number

        runs = runs_experiment
        script_dir = os.path.dirname(__file__)
        data_dir = os.path.join(script_dir, "./physionet.org/files/eegmmidb/1.0.0/")
        files = eegbci.load_data(subject, runs, data_dir)

        raws = [read_raw_edf(f, preload=True) for f in files]
        raw_obj = concatenate_raws(raws)
        eegbci.standardize(raw_obj)
        montage = make_standard_montage("standard_1005")
        raw_obj.set_montage(montage)

        return raw_obj
    except Exception as e:
        print(e)
        exit()


def filter_edf(raw):
    """Take the Raw object  and apply filter to kept the right information
    return the raw object filtered

    Parameter:
    raw Raw object
    """
    raw = raw.filter(8.0, 40.0, fir_design="firwin", skip_by_annotation="edge")
    return raw


def define_epochs(raw):
    """
    We need to define epoch ( little piece of data containing event that is define in the file )  and get their data and their tag
    to use them to process a csp on it

    return data ndarray (n_epoch, n_channels ,n_times)  and labels the tag of each epoch ndarray (n_epoch)

    Parameter:
    Raw object
    """
    picks = pick_types(
        raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
    )

    events, event_id = events_from_annotations(raw, dict(T1=1, T2=2))
    tmin, tmax = -0.1, 4.0
    epochs = Epochs(
        raw,
        events,
        event_id,
        tmin=tmin,
        tmax=tmax,
        proj=True,
        picks=picks,
        preload=True,
        baseline=None,
    )
    print(f"Number of epochs: {len(epochs)}")
    epochs_train = epochs.copy()

    labels = epochs.events[:, -1]
    epochs_data_train = epochs_train.get_data(copy=False)

    return epochs_data_train, labels


def preprocessing_data(n_task, n_subject):
    """Preprocess the data
    1- create dict of all subject with their experiments
    2- filter the frequency
    3-  create epoch
    return  X for data (n_epoch, n-channels, n_times) and y the tag of each epoch (ndarray of n_epoch)

    Parameters:
    n_task list of number of task
    n_subject int number of subject
    """
    raw = concatenate_edf(subject_number=n_subject, runs_experiment=n_task)
    raw = filter_edf(raw)
    new_X, new_Y = define_epochs(raw)

    return new_X, new_Y
