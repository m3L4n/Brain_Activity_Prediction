import os
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne import Epochs, pick_types
from mne import (
    annotations_from_events,
    events_from_annotations,
)


def concatenate_edf(subject_number: int, runs_experiment: list):
    """
    take the number of the subject and the experiments to test
    it will load from the directory physionet.org placed a the root of the project all the .edf
    and concatenate all the .edf  together to get one file of egg
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
        events, _ = events_from_annotations(raw_obj, event_id=dict(T1=1, T2=2))
        annot_from_events = annotations_from_events(
            events=events,
            sfreq=raw_obj.info["sfreq"],
        )
        raw_obj.set_annotations(annot_from_events)
        return raw_obj
    except Exception as e:
        print(e)
        exit()


def filter_edf(raw):
    """
    take the raw and apply filter to the frequency kept
    """
    raw = raw.filter(8.0, 40.0, fir_design="firwin", skip_by_annotation="edge")
    return raw


def define_epochs(raw):
    picks = pick_types(
        raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
    )

    events, event_id = events_from_annotations(raw)
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
    return a X for data (n_epoch, n-channels, n_times) and y the tag of each epoch vector of n_epoch
    """
    raw = concatenate_edf(subject_number=n_subject, runs_experiment=n_task)
    raw = filter_edf(raw)
    new_X, new_Y = define_epochs(raw)

    return new_X, new_Y
