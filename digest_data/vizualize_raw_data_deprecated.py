import mne
import matplotlib.pyplot as plt
import os

import numpy as np


def get_directory_data():
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(
        script_dir, '../physionet.org/files/eegmmidb/1.0.0')
    return data_dir


def get_list_edf_file(directory, raw_list, index, end):
    """
    """
    items = os.listdir(directory)
    for item in items:
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            print(f"Répertoire : {item_path}")
            if (index > end):
                return
            index += 1
            get_list_edf_file(item_path, raw_list, index, end)
        else:
            if item.endswith(".edf"):
                print(item)
                file_path = os.path.join(directory, item)
                raw = mne.io.read_raw_edf(file_path, preload=True)
                # ts different from the frequency  of the signal, its represent the each sampled at 160 samples per second
                if raw.info['sfreq'] != 160:
                    raw.resample(160)
                raw_list.append(raw)
                print(f"Fichier : {item_path}")


def concatenate_raws(raw_list):
    """
    concatenate all the file edf together
    """
    min_sfreq = min([raw.info['sfreq'] for raw in raw_list])
    print(min_sfreq)
    raw_combined = mne.concatenate_raws(raw_list)
    return raw_combined


# def filter_frequency(raw_combined):
#     """
#     calcul of the PSD ( puissance spectrale density) to keep only the
#     frequency that important ( not noise for example)
#     so we will keep only this frequency that represent emotion for example
#     alpha onde represent the state of the relaxation
#     get only this frequency
#         Bande Delta (0.5-4 Hz) : Généralement associée au sommeil profond.
    # Bande Theta(4-8 Hz): Souvent associée à la somnolence et à la relaxation.
    # Bande Alpha(8-13 Hz): Typiquement liée à l'état de relaxation, surtout avec les yeux fermés.
    # Bande Beta(13-30 Hz): Associée à l'activité cognitive, l'éveil et la concentration.
    # Bande Gamma(30-100 Hz): Liée à des processus cognitifs supérieurs comme l'attention et la mémoire de travail.
    # Le pic autour de 50 Hz est probablement un bruit de l'alimentation électrique(50 Hz est la fréquence du courant alternatif en Europe, 60 Hz aux États-Unis).
    # Les hautes fréquences au-dessus de 40 Hz peuvent contenir du bruit musculaire ou d'autres artefacts non liés au signal EEG.
#     """
#     data = raw_combined.get_data()
#     data_length = len(raw_combined.times)
#     n_fft = min(data_length, 2048)
#     psds, freqs = mne.time_frequency.psd_array_welch(
#         data, 160, fmin=0.5, fmax=80., n_fft=n_fft)
#     # n_fft is for the Elle décompose un signal en ses composantes de fréquence.
#     # so its elevate its will increase the complexity of the calcul but  its will more sucessfull
#     # doit etre inférieur ou égal à la longueur du segment du signal analysé its important for distinctive the differant  frequency
#     return psds, freqs


# def show_PSD(freqs, psds):
#     """
#     """
#     # psds_db = 10 * np.log10(psds)

#     psds_db = 10 * np.log10(psds)

#     plt.figure(figsize=(10, 5))
#     plt.plot(freqs, psds_db.T)
#     plt.title('Densité spectrale de puissance (PSD)')
#     plt.xlabel('Fréquence (Hz)')
#     plt.ylabel('Puissance (uV^2/Hz) [dB]')
#     plt.show()


# def plot_eeg(raw_combined):
#     # Obtenir les données brutes
#     data, times = raw_combined[:, :]

#     # Tracer les signaux EEG dans le domaine temporel
#     plt.figure(figsize=(15, 5))
#     plt.plot(times, data.T * 1e6)  # Convertir en microvolts
#     plt.xlabel('Temps (s)')
#     plt.ylabel('Amplitude (µV)')
#     plt.title('Signaux EEG bruts')
#     plt.show()


def main():
    raw_list = []
    directory = get_directory_data()

    get_list_edf_file(directory, raw_list, 0, 1)
    raw_combined = concatenate_raws(raw_list)
    # data = raw_combined.get_data()
    events = mne.find_events(raw_combined)

    epochs = mne.Epochs(raw_combined, events, tmin=-1, tmax=1,
                        baseline=(None, 0), event_id=1)
    psd, freqs = mne.time_frequency.psd_multitaper(
        epochs, fmin=2, fmax=40, n_jobs=1)

    psd_mean = psd.mean(axis=0)
    mne.viz.plot_psd(psd_mean, freqs)
    # raw_combined.plot()
    # # # Calculer la PSD
    # psds, freqs = mne.time_frequency.psd_array_welch(
    #     data, sfreq=160, fmin=0.5, fmax=80.0, n_fft=n_fft)

    # # Tracer les courbes pour les 64 canaux
    # plt.figure(figsize=(15, 10))
    # for i in range(64):
    #     plt.plot(freqs, psds[i], label=f'Channel {i+1}')

    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power Spectral Density (dB/Hz)')
    # plt.title('Power Spectral Density (PSD) of EEG for 64 Channels')
    # plt.legend(ncol=4)
    # plt.grid(True)
    # plt.show()

    # print(raw_combined)
    # raw_combined.compute_psd().plot()
    # # raw_combined.plot(duration=10, n_channels=64)
    plt.show()
    # print(raw_list, len(raw_list))


if __name__ == "__main__":
    main()
