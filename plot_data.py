import matplotlib.pyplot as plt


def plot_data(
    raw,
):
    raw.plot()
    raw.compute_psd().plot()
    plt.show()
    print("here he plot data", raw)
