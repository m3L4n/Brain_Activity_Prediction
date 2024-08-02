from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.pipeline import Pipeline


from CSP.csp import CSP


def set_up_pipeline():
    """
    set up the pipeline with our algorithmn CSP and classifier LDA
    """
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=6)
    clf = Pipeline([("CSP", csp), ("LDA", lda)])
    return clf
