
'''
Author: Nyasha Masamba
Date: 3rd September 2023

Purpose of program
-------------------
End-to-end script for testing and logging the functionpredicting customer churn.
Includes data import, EDA, feature engineering and model training functions.
'''

def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        df = pd.read_csv("./data/bank_data.csv")
        perform_eda(df)
        assert(os.path.exists('images/eda/churn_hist.png'))
        assert(os.path.exists('images/eda/customer_age_hist.png'))
        assert(os.path.exists('images/eda/marital_status_bar.png'))
        assert(os.path.exists('images/eda/total_transactions_hist.png'))
        assert(os.path.exists('images/eda/heatmap.png'))
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_eda: The EDA plots have not been saved.")
        raise err
        


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        df = pd.read_csv("./data/bank_data.csv")
        df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
        X, y = encoder_helper(df, ['Gender', 'Education_Level', 'Marital_Status',
                                      'Income_Category', 'Card_Category'])
        assert(isinstance(X, pd.DataFrame))
        assert(isinstance(y, pd.Series))
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper: Encoded data not available.")
        raise err
        
    

def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        df = pd.read_csv("./data/bank_data.csv")
        df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
        X, y = cls.encoder_helper(df, ['Gender', 'Education_Level', 'Marital_Status',
                                      'Income_Category', 'Card_Category'])
        X_train_set, X_test_set, y_train_set, y_test_set = perform_feature_engineering(
    X, y)
        assert(isinstance(X_train_set, pd.DataFrame))
        assert(isinstance(X_test_set, pd.DataFrame))
        assert(isinstance(y_train_set, pd.Series))
        assert(isinstance(y_test_set, pd.Series))
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper: Training features not available.")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        df = pd.read_csv("./data/bank_data.csv")
        df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
        X, y = cls.encoder_helper(df, ['Gender', 'Education_Level', 'Marital_Status',
                                      'Income_Category', 'Card_Category'])
        X_train_set, X_test_set, y_train_set, y_test_set = cls.perform_feature_engineering(
    X, y)
        y_train_preds_lr_set, y_train_preds_rf_set, y_test_preds_lr_set, y_test_preds_rf_set, saved_rf, saved_lr = train_models(
    X_train_set, X_test_set, y_train_set, y_test_set)
        assert(os.path.exists('./models/rfc_model.pkl'))
        assert(os.path.exists('./models/logistic_model.pkl'))
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_models: Model training was not completed.")
        raise err


if __name__ == "__main__":
    
    # import libraries
    import os
    import logging
    import pandas as pd
    import churn_library as cls
    
    # set up log file
    logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

    # run tests
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)








