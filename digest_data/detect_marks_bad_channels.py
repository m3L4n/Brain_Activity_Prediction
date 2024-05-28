import numpy as np


def detect_marks_bad_channels(raw):
    """
    detect channels that is not normal, for example a flat frequency of the channels
    """
    variances = np.var(raw.get_data(), axis=1)
    threshold = np.mean(variances) + 3 * np.std(variances)
    bad_channels = [raw.ch_names[i]
                    for i in range(len(variances)) if variances[i] > threshold]
    raw.info['bads'] = bad_channels
