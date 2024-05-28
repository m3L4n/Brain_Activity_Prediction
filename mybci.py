import sys

from digest_data.define_epochs import define_epochs
from digest_data.montage_raw import montage_raw
from digest_data.filter_frequency import filter_edf
from digest_data.concatenate_raw1 import concatenate_edf
from digest_data.detect_marks_bad_channels import detect_marks_bad_channels
from plot_data.plot_data import plot_data
from predict_data.predict_data import predict_data
from train_data.train_data import train_data


def check_argument(nbr_subj, str_runs):
    try:
        nbr_subj = int(nbr_subj)
        assert isinstance(
            str_runs, str), "Error str_runs must be a string"
        array_runs = str_runs.split(" ")
        array_runs = [int(x) for x in array_runs]
        return nbr_subj, array_runs
    except Exception as e:
        print("Error", e)
        raise e


def get_mode(str):
    if str == 'predict':
        return 2
    elif str == 'train':
        return 1
    elif str == 'plot':
        return 0
    return -1


def set_up_data(subject_index, runs_index, index_mode):
    """
    preprocessing of the data before to go to the predict or the train
    """
    try:
        raw = concatenate_edf(subject_index, runs_index)
        if index_mode == 0:  # for plot we have to plot raw data and filtered data
            return raw, None, None
        # detect_marks_bad_channels(raw)
        filter_edf(raw)
        raw = montage_raw(raw)
        epoch_data_train, labels = define_epochs(raw)

        return raw, epoch_data_train, labels
    except Exception as e:
        print('error initialize data', e)


def main(argv):
    list_function_mode = [plot_data, train_data, predict_data]
    try:
        if (len(argv) == 1):
            print("Not ready for the test validation")
        elif len(argv) == 4:
            subject_index, runs_index = check_argument(argv[1], argv[2])
            index_func_mode = get_mode(argv[3])
            assert not index_func_mode == -1, "Error the mode need to be  predict| train | plot"
            raw, epoch_data_train, labels = set_up_data(
                subject_index, runs_index, index_func_mode)
            list_function_mode[index_func_mode](raw, epoch_data_train, labels)
        else:
            print("Error\nUsage: python3 ./mbcy.py for test validation or ./mybci.py index_subject ' index runs seprate by space' predit|train ")
    except Exception as e:
        print(e)
        exit()


if __name__ == "__main__":
    main(sys.argv)
