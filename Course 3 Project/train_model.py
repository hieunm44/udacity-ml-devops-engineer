# Script to train machine learning model.
from sklearn.model_selection import train_test_split
# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, compute_slice_metrics
import pickle


# Add code to load in the data.
data = pd.read_csv('data/census_cleaned.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label='salary', training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
pickle.dump(model, open('model/model.pkl', 'wb'))
pickle.dump(encoder, open("model/encoder.pkl", 'wb'))
pickle.dump(lb, open('model/lb.pkl', 'wb'))

# Model evaluation
y_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
print(f"precision: {precision}, recall: {recall}, fbeta: {fbeta}")

# Evaluation on data slices
with open('slice_output.txt', 'w') as f:
    for cat_feature in cat_features:
        slice_metrics = compute_slice_metrics(test, cat_feature, y_test, y_pred)
        for key, value in slice_metrics.items():
            f.write(f"{cat_feature} = {key}: {value}\n")