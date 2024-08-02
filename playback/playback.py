import numpy as np
import time


def playback_reading(epoch_ndarray, pipeline, labels, delay=1):
    print("\033[0;34;1m epoch nb : [prediction] [truth] [equal?] \033[0;34;1m")
    for idx, (epoch, truth_label) in enumerate(zip(epoch_ndarray, labels)):
        epoch_data = epoch[np.newaxis, :]
        y_new_pred = pipeline.predict(epoch_data)
        color = "31;1"
        if y_new_pred == truth_label:
            color = "32;1"
        print(
            f"\033[0;{color}m epoch {idx}: {y_new_pred} [{truth_label}] {y_new_pred == truth_label} \033[0;{color}m"
        )
        time.sleep(delay)
