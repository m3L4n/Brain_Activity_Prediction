def plot_data(raw,  epochs_data_train, labels):
    raw.plot()
    raw.compute_psd().plot()
    print("here he plot data", raw)
