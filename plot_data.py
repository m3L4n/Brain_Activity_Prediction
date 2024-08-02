import matplotlib.pyplot as plt

from preprocessing_data import concatenate_edf, filter_edf


def plot_data(n_subject, n_tasks):
    """Plot data for n subject and n tasks before and after preprocessing

    Parameters:
    n_task list of number of task
    n_subject int number of subject
    """
    try:
        raw = concatenate_edf(n_subject, n_tasks)
        raw.plot()
        raw.compute_psd().plot()
        plt.show()
        raw = filter_edf(raw)
        raw.plot()
        raw.compute_psd().plot()
        raw.compute_psd().plot(average=True, picks="data", exclude="bads")
        plt.show()

    except Exception as e:
        print(f"{type(e).__name__}", e)
