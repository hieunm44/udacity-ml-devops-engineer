# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model was created by Nguyen Minh Hieu (HieuNM44).

The model is a `RandomForestClassifier` from library `sklearn.ensemble`.

## Intended Use
The model predicts whether a person's income exceeds $50K/year based on their census data.

## Training Data
Data file location: `data/census.csv`.

Details about the data set can be found [here](https://archive.ics.uci.edu/ml/datasets/census+income)

Extraction was done by Barry Becker from the 1994 Census database.

Prediction task is to determine whether a person makes over 50K a year.

## Evaluation Data
The data set is cleaned, preprocessed and split into trainging and evaluation data with a evaluation ratio of 20%.

## Metrics
Performance metrics of the trained model:
* precision: 0.7362385321100917
* recall: 0.622093023255814
* fbeta: 0.6743697478991596

## Ethical Considerations
The model is not biased towards any specific group of people.

## Caveats and Recommendations
The dataset should not be used as a statistical representation of the population. It is recommended to use the dataset for research purposes only.