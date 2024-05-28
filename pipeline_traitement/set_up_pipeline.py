from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
from sklearn.pipeline import Pipeline

# TO DO remplace CSP by our implementation and  use
# base estomator et classifier mixin


def set_up_pipeline():
    """
    set up the pipeline with our algorithmn CSP and classifier LDA
    """
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    clf = Pipeline([("CSP", csp), ("LDA", lda)])
    return clf
