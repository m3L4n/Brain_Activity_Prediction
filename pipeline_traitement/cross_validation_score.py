from sklearn.model_selection import cross_val_score, ShuffleSplit


def cross_validation_score(
    pipeline,
    epoch_data,
    labels_event,
):
    cv = ShuffleSplit(5, test_size=0.2, random_state=42)
    scores = cross_val_score(pipeline, epoch_data, labels_event, cv=cv, n_jobs=None)
    return scores
