import numpy as np

from pipeline_traitement.cross_validation_score import cross_validation_score
from preprocessing_data import preprocessing_data
from pipeline_traitement.set_up_pipeline import set_up_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def test_all_data():
    """Script that will launch the training for all subject the 6 experiments
    1- open and close left or right fist,
    2- imagine opening and closing left or right fist
    3- open and close both fists or both feet
    4- imagine opening and closing both fists or both feet
    5- open - close /imagine open-close left or right fist
    6- open - close /imagine open-close both fists or both feet
    and print them  and give the mean of each experiments and the global mean
    """
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
    task_prediction_accuraccy = {}
    for i in range(1, number_subject):
        task_prediction[i] = {}
        task_prediction_accuraccy[i] = {}
        for key, value in task_dict.items():
            task_prediction[i][key] = []
            task_prediction_accuraccy[i][key] = []

    for subject in range(1, number_subject):
        for task_id, runs in task_dict.items():
            data, labels = preprocessing_data(runs, subject)
            clf = set_up_pipeline()
            scores = cross_validation_score(clf, data, labels)
            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            task_prediction[subject][task_id].append(np.mean(scores))
            task_prediction_accuraccy[subject][task_id].append(accuracy_score(y_test, y_pred))
    task_res_dict = {}
    accuracy_res_dict = {}
    pred_final = []
    pred_final_cross = []
    for i in range(1, 7):
        task_res_dict[i] = []
        accuracy_res_dict[i] = []

# Computing cross validation score on every subject
    for subject, result in task_prediction.items():
        for exp_id, result_exp in result.items():
            print(f"Subject {subject} experiment {exp_id} accuracy  cross validation{result_exp[0]}")
            task_res_dict[exp_id].append(result_exp[0])

# Computing accuracy score on every subject
    for subject, result in task_prediction_accuraccy.items():
        for exp_id, result_exp in result.items():
            print(f"Subject {subject} experiment {exp_id} accuracy (accuracy_score) {result_exp[0]}")
            accuracy_res_dict[exp_id].append(result_exp[0])
    
    # cross validation score mean
    for key, value in task_res_dict.items():
        pred_final_cross.append(np.mean(value))
        print(f"Experiment {key} accuracy (cross validation score){np.mean(value)}")

    # accuracy score mean
    for key, value in accuracy_res_dict.items():
        pred_final.append(np.mean(value))
        print(f"Experiment {key} accuracy ((accuracy score) {np.mean(value)}")

    print("Mean accuracy score of the 6 experiment", np.mean(pred_final))
    print("Mean cross validation score of the 6 experiment", np.mean(pred_final_cross))


def main():
    test_all_data()


if __name__ == "__main__":
    main()
