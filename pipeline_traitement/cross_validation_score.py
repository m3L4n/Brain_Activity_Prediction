from sklearn.model_selection import cross_val_score, ShuffleSplit


def cross_validation_score(
    pipeline,
    epoch_data,
    labels_event,
):
    """Wrapper of cross_val_score with shuffle split already in it
    Return array of len 5(nbr of cv ) with the result of the accuracy of the pipeline

    Parameters:
    pipeline : sklearn pipeline
    epoch_data : data  ( n_epoch, n_channels, n_times)
    """
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    scores = cross_val_score(pipeline, epoch_data, labels_event, cv=cv, n_jobs=10)
    return scores
