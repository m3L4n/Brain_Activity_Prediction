import os
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
import sys


def concatenate_edf(subject_number: int, runs_experiment: list):
    """
    take the number of the subject and the experiments to test
    it will load from the directory physionet.org placed a the root of the project all the .edf 
    and concatenate all the .edf  together to get one file of egg
    """
    print(type(subject_number), type(runs_experiment))
    try:
        assert isinstance(
            subject_number, int), "Error subject number has to be int"
        assert isinstance(
            runs_experiment, list), "Error runs_experiment has to be list"
        assert all([int(x) for x in runs_experiment]
                   ), "Error runs_experiment has to be a list of int"

        subject = subject_number

        runs = runs_experiment
        script_dir = os.path.dirname(__file__)
        data_dir = os.path.join(
            script_dir, '../physionet.org/files/eegmmidb/1.0.0/')
        files = eegbci.load_data(
            subject, runs, data_dir)

        raws = [read_raw_edf(f, preload=True) for f in files]
        raw_obj = concatenate_raws(raws)
        eegbci.standardize(raw_obj)  # Important because it set channel names
        return raw_obj
    except Exception as e:
        print(e)
        exit()


def main(argv):
    try:
        nbr_subj = int(argv[1])
        str_runs = argv[2]
        array_runs = str_runs.split(" ")
        array_runs = [int(x) for x in array_runs]
        print(type(array_runs))
        concatenate_edf(nbr_subj, array_runs)
    except Exception as e:
        print("error", e)
        exit()


if __name__ == "__main__":
    main(sys.argv)
