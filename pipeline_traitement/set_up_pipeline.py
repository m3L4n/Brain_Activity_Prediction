from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.pipeline import Pipeline


from CSP.csp import CSP


def set_up_pipeline():
    """Set up the pipeline with our algorithm CSP and classifier LDA
    Return the set up pipeline
    """
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=6)
    clf = Pipeline([("CSP", csp), ("LDA", lda)])
    return clf
