import mne


def montage_raw(raw):
    """
    take the raw edf and give it the right montage ( location of channels)
    because it will help to better preproccess our raw
    """

    montage_1 = mne.channels.make_standard_montage("standard_1005")
    raw = raw.set_montage(
        montage_1)
    raw.set_eeg_reference(projection=True)
    return raw
