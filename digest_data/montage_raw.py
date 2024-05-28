import mne


def montage_raw(raw):
    """
    take the raw edf and give it the right montage ( location of channels)
    because it will help to better preproccess our raw
    """

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
