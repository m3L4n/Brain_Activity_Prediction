from sklearn.model_selection import cross_val_score,


def cross_validation_score(pipeline, epoch_data, labels_event, cv, nb_jobs=None):

    scores = cross_val_score(pipeline, epoch_data,
                             labels_event, cv=cv, n_jobs=None)
    return scores
