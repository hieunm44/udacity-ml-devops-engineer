# Predict Customer Churn with Clean Code

Course 1 - Clean Code Principles.

## Project Description
This is the implementation for Churn Prediction using customer's data.

## Files and data description
Below is the overview of the files and data present in the root directory:
.
├── Guide.ipynb          # Given: Getting started and troubleshooting tips
├── churn_notebook.ipynb # Code draft in notebook
├── churn_library.py     # Functions to be used
├── churn_script_logging_and_tests.py # Tests and logs
├── README.md            # 
├── data                 # Bank data
│   └── bank_data.csv
├── images               # Store EDA results 
│   ├── eda
│   └── results
├── logs				 # Store logs
└── models               # Store models

## Running Files
### 1. Set up
Install dependencies:
```
python -m pip install -r requirements_py3.6.txt
```
### 2. Run churn_library to process data, perform eda, train models and create result images
```
python churn_library.py
```
Then eda images and results images are crated in 'images/'
### 3. Run logging and tests
```
python churn_script_logging_and_tests.py
```
Then a logging file is created at 'logs/churn_library.log'



