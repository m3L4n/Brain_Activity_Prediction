import numpy as np

from preprocessing_data import preprocessing_data
from pipeline_traitement.set_up_pipeline import set_up_pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score


def test_all_data():
    number_subject = 110
    task_dict = {
        1: [3, 7, 11],
        2: [4, 8, 12],
        3: [5, 9, 13],
        4: [6, 10, 14],
        5: [3, 7, 11, 4, 8, 12],
        6: [5, 9, 12, 6, 10, 14],
    }
    task_prediction = {}
    for i in range(1, number_subject):
        task_prediction[i] = {}
        for key, value in task_dict.items():
            task_prediction[i][key] = []

    for subject in range(1, number_subject):
        for task_id, runs in task_dict.items():
            data, labels = preprocessing_data(runs, subject)
            clf = set_up_pipeline()
            cv = ShuffleSplit(5, test_size=0.2, random_state=42)
            scores = cross_val_score(clf, data, labels, cv=cv, n_jobs=None)
            clf.fit(data, labels)

            task_prediction[subject][task_id].append(np.mean(scores))
    task_res_dict = {}
    for i in range(1, 7):
        task_res_dict[i] = []
    for subject, result in task_prediction.items():

        print("result task", task_prediction, result)
        for exp_id, result_exp in result.items():
            print("result", result_exp, exp_id)
            print(f"Subject {subject} experiment {exp_id} accuracy {result_exp[0]}")
            task_res_dict[exp_id].append(result_exp[0])
    for key, value in task_res_dict.items():
        print(f"Experiment {key} accuracy{np.mean(value)}")


def main():
    test_all_data()


if __name__ == "__main__":
    main()
