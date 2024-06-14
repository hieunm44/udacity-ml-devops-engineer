"""
Module: churn_library

Author: Hieu Nguyen Minh
Date: April 6, 2023

"""


# import libraries
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
        pth: a path to the csv
    output:
        dataframe: pandas dataframe
    '''

    try:
        dataframe = pd.read_csv(pth)
    except FileNotFoundError:
        return 'The file does not exist.'

    return dataframe


def perform_eda(dataframe):
    '''
    perform eda on dataframe and save figures to images folder
    input:
        dataframe: pandas dataframe

    output:
        None
    '''

    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    dataframe['Churn'].hist()
    plt.savefig('images/eda/churn_histogram.png')

    plt.figure(figsize=(20, 10))
    dataframe['Customer_Age'].hist()
    plt.savefig('images/eda/Customer_Age_histogram.png')

    plt.figure(figsize=(20, 10))
    dataframe.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('images/eda/Martial_Status_value_counts.png')

    plt.figure(figsize=(20, 10))
    sns.histplot(dataframe['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('images/eda/Total_Trans_Ct_density.png')

    plt.figure(figsize=(20, 10))
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('images/eda/df_correlation_matrix.png')


def encoder_helper(dataframe, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
        dataframe: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name
                  [optional argument that could be used for naming variables or index y column]

    output:
        dataframe: pandas dataframe with new columns for category_lst
    '''

    for col in category_lst:
        churn_mean = dataframe.groupby(col).mean()['Churn']
        dataframe[col + '_' + response] = dataframe[col].map(churn_mean)

    return dataframe


def perform_feature_engineering(dataframe, response='Churn'):
    '''
    input:
        dataframe: pandas dataframe
        response: string of response name
                  [optional argument that could be usedfor naming variables or index y column]

    output:
        x_train: X training data
        x_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''

    y_label = dataframe[response]
    x_data = pd.DataFrame()

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    x_data[keep_cols] = dataframe[keep_cols]
    # print('\nSome first rows of X:\n', X.head())

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_label, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
        y_train: training response values
        y_test:  test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest

    output:
        None
    '''

    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('images/results/rfc_classification_report.png')

    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('images/results/lr_classification_report.png')


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
        model: model object containing feature_importances_
        x_data: pandas dataframe of X values
        output_pth: path to store the figure

    output:
        None
    '''

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth)


def roc_plot(model, x_data, y_label, output_pth):
    '''
    plot ROC and save images
    input:
        model: model object containing feature_importances_
        x_data: pandas dataframe of X values
        y_label: pandas series of y values
        output_pth: path to store the figure

    output:
        None
    '''

    plt.figure(figsize=(5, 5))
    plot_roc_curve(model, x_data, y_label)
    plt.savefig(output_pth)


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
        x_train: X training data
        x_test: X testing data
        y_train: y training data
        y_test: y testing data
    output:
        None
    '''

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [20, 50],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4],
        'criterion': ['gini']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, 'models/rfc_model.pkl')
    joblib.dump(lrc, 'models/logistic_model.pkl')

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    feature_importance_plot(
        cv_rfc.best_estimator_,
        x_test,
        'images/results/rfc_feature_importance.png')

    roc_plot(
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        'images/results/rfc_roc.png')
    roc_plot(lrc, x_test, y_test, 'images/results/lr_roc.png')


if __name__ == '__main__':
    df = import_data('data/bank_data.csv')
    # perform_eda(df)
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    churn_mean_df = encoder_helper(df, cat_columns, response='Churn')
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        churn_mean_df)
    train_models(x_train, x_test, y_train, y_test)

    rfc_model = joblib.load('models/rfc_model.pkl')
    lr_model = joblib.load('models/logistic_model.pkl')
