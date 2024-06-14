"""
Module: churn_script_logging_and_tests

Author: Hieu Nguyen Minh
Date: April 6, 2023

"""

import logging
import glob
import pandas as pd
from churn_library import import_data, perform_eda, encoder_helper, \
                          perform_feature_engineering, train_models
import constants

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import
    - this example is completed for you to assist with the other test functions
    '''
    try:
        dataframe = import_data(constants.DATA_PATH)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''

    logging.info('Testing perform_eda function...')
    dataframe = import_data(constants.DATA_PATH)
    perform_eda(dataframe)
    eda_images = glob.glob('images/eda/*.png')

    for image_path in constants.EDA_IMAGE_PATHS:
        try:
            assert image_path in eda_images
            logging.info('SUCCESS: {} has been generated.'.format(image_path))
        except AssertionError:
            logging.error(
                'ERROR: {} has not been generated.'.format(image_path))


def test_encoder_helper():
    '''
    test encoder helper
    '''

    logging.info('Testing encoder_helper function...')

    dataframe = import_data(constants.DATA_PATH)
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    churn_mean_df = encoder_helper(
        dataframe,
        constants.CAT_COLUMNS,
        response='Churn')

    for col in constants.CAT_COLUMNS:
        try:
            assert col + '_Churn' in churn_mean_df.columns
            logging.info(
                'SUCCESS: Column {} has been generated.'.format(
                    col + '_Churn'))
        except AssertionError:
            logging.error(
                'ERROR: Column {} has not been generated.'.format(
                    col + '_Churn'))


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''

    logging.info('Testing perform_feature_engineering function...')

    dataframe = import_data(constants.DATA_PATH)
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    churn_mean_df = encoder_helper(
        dataframe,
        constants.CAT_COLUMNS,
        response='Churn')

    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            churn_mean_df)
        logging.info('SUCCESS: Training and test sets have been generated.')
    except BaseException:
        logging.error('ERROR: Cannot generated training and test sets.')

    try:
        assert isinstance(x_train, pd.DataFrame)
        assert isinstance(x_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        logging.info(
            'SUCCESS: All training and test data have correct data types.')
    except AssertionError:
        logging.error(
            'ERROR: Training and test data do not have correct data types.')


def test_train_models():
    '''
    test train_models
    '''

    logging.info('Testing train_models function...')

    dataframe = import_data(constants.DATA_PATH)
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    churn_mean_df = encoder_helper(dataframe, constants.CAT_COLUMNS, response='Churn')

    x_train, x_test, y_train, y_test = perform_feature_engineering(
        churn_mean_df)
    train_models(x_train, x_test, y_train, y_test)
    try:
        train_models(x_train, x_test, y_train, y_test)
        logging.info('SUCCESS: Models have been trained')
    except BaseException:
        logging.error('ERROR: Cannot train models')

    result_images = glob.glob('images/results/*.png')
    for image_path in constants.RES_IMAGE_PATHS:
        try:
            assert image_path in result_images
            logging.info('SUCCESS: {} has been generated.'.format(image_path))
        except AssertionError:
            logging.error(
                'ERROR: {} has not been generated'.format(image_path))


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
