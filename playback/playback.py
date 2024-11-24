import numpy as np
import time
from sklearn.metrics import accuracy_score


def playback_reading(epoch_ndarray, pipeline, labels, delay=1):
    """
    Simulae real time data, it send epoch per epoch each one second
    and predict each data receive and print them

    Parameters:
    epoch_ndarray : epoch data (n_epoch, n_channels, n_time)
    pipeline : sklearn pipeline
    label :tag of all epoch ndarray (n_epoch)
    """

    red = "31;1"
    green = "32;1"
    blue = "34;1"
    pred_final = []
    print(f"\033[0;{blue}m epoch nb : [prediction] [truth] [equal?] \033[0;{blue}m")
    for idx, (epoch, truth_label) in enumerate(zip(epoch_ndarray, labels)):
        epoch_data = epoch[np.newaxis, :]
        y_new_pred = pipeline.predict(epoch_data)
        pred_final.append(y_new_pred)
        color = red
        if y_new_pred == truth_label:
            color = green
        print(
            f"\033[0;{color}m epoch {idx}: {y_new_pred} [{truth_label}] {y_new_pred == truth_label} \033[0;{color}m"
        )
        time.sleep(delay)
    print(f"Accuracy {accuracy_score(labels, pred_final)}")
