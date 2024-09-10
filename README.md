# Total perspective vortex

> Brain computer interface with machine learning based on
electoencephalographic data

This repository contains a comprehensive pipeline designed to classify EEG data using a Common Spatial Pattern (CSP)  from scratch in Python and Linear Discriminant Analysis (LDA). The objective is to predict whether a subject is performing or imagining Movement A or Movement B.

## Workflow Overview

The workflow of our data processing pipeline is as follows:

    - EDF File: The EEG data is initially collected and stored in an EDF (European Data Format) file.
    - Concatenate EDF: Multiple EDF files are concatenated to form a single continuous dataset for analysis.
    - Montage: A montage is applied to the EEG data to map the electrode positions to standard locations.
    - Filter Right Frequency: The EEG data is filtered to retain the frequencies relevant for motor imagery tasks.
    - Define Epoch: The continuous EEG data is segmented into epochs, each representing a specific time window of the EEG signal.
    - Pipeline Operations:
        - Fit/Train: The epochs and their corresponding labels are used to fit and train the CSP and LDA classifiers.
        - Transform/Predict: The trained model is then used to transform new EEG data and predict whether the subject is performing or imagining Movement A or Movement B


Usage :
```
python -m venv [path] 
source [path]/bin/activate
pip install -r requirements.txt
python ./mybci to train all subject
python ./mybci --action [plot | train | predict] --subject [id subject] -- tasks [N N N N N]
N between 3 and 14
and id subject 1 to 109
```

