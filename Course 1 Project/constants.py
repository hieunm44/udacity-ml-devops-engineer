"""
Module: constants

Author: Hieu Nguyen Minh
Date: April 6, 2023

"""


DATA_PATH = 'data/bank_data.csv'

EDA_IMAGE_PATHS = ['images/eda/churn_histogram.png',
                   'images/eda/Customer_Age_histogram.png',
                   'images/eda/Martial_Status_value_counts.png',
                   'images/eda/Total_Trans_Ct_density.png',
                   'images/eda/df_correlation_matrix.png']

CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

RES_IMAGE_PATHS = ['images/results/rfc_classification_report.png',
                   'images/results/lr_classification_report.png',
                   'images/results/rfc_feature_importance.png',
                   'images/results/rfc_roc.png',
                   'images/results/lr_roc.png']
