import mne
import matplotlib.pyplot as plt
import os
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
import numpy as np


def get_directory_data():
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(
        script_dir, '../physionet.org/files/eegmmidb/1.0.0/')
    return data_dir


def get_concatenate_edf(index_max=10):

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
    raw.plot_psd()
    plt.show()
    fig = raw.compute_psd(tmax=np.inf, fmax=80).plot(
        average=True, amplitude=False, picks="data", exclude="bads"
    )
    plt.show()


def show_raw_plot(raw):
    raw.plot_psd()  # to see the psd  with the different frequency
    raw.plot()
    plt.show()


def main():
    raw = get_concatenate_edf()
    show_raw_plot(raw)
    filter_frequency(raw)

    plt.show()


if __name__ == "__main__":
    main()
