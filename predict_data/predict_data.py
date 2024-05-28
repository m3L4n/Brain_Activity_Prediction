import os
import joblib


def predict_data(raw, epochs_data, labels):
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, "../models/pipeline.pkl")
    pipeline = joblib.load(data_dir)
    y_new_pred = pipeline.predict(epochs_data)
    # scores = cross_val_score(pipeline, epochs_data,
    #                          labels, cv=8, n_jobs=None)
    correct = 0
    total = len(y_new_pred)
    for i, (pred, truth) in enumerate(zip(y_new_pred, labels)):
        print(f"epoch {i:02d}: [{pred}] [{truth}] {pred == truth}")
        if pred == truth:
            correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
