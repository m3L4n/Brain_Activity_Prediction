def filter_edf(raw):
    """
    take the raw and apply filter to the frequency kept
    """
    raw.filter(8., 40., fir_design='firwin', skip_by_annotation='edge')
    # raw.notch_filter(freqs=50.0, notch_widths=1.0)
