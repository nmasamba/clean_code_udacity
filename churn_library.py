# library doc string
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

# import libraries
import os
os.environ['QT_QPA_PLATFORM']='offscreen'



def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # churn_plot
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    df['Churn'].hist()
    plt.savefig('images/eda/churn_hist.png')
    
    # customer_age_plot
    df['Customer_Age'].hist()
    plt.savefig('images/eda/customer_age_hist.png')
    
    # marital_status_plot
    plt.figure(figsize=(20,10)) 
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('images/eda/marital_status_bar.png', dpi='figure')
    
    # total_transactions_plot
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('images/eda/total_transactions_hist.png')
    
    # heatmap
    plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig('images/eda/heatmap.png', dpi='figure')
    
    


def encoder_helper(df, category_lst, response='y'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            X: pandas dataframe with new encoded features
            response: target/y column
    '''
    # encode categorical features based on categorical column list
    for categorical_col in category_lst:
        category_lst = []
        category_groups = df.groupby(categorical_col).mean()['Churn']
        for val in df[categorical_col]:
            category_lst.append(category_groups.loc[val])
        df[categorical_col+'_Churn'] = category_lst
    
    # get response series and keep specific features in the dataframe
    response = df['Churn']
    X = pd.DataFrame()
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']
    X[keep_cols] = df[keep_cols]
    
    return X, response

def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: target/y column

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(df, response, test_size= 0.3, random_state=42)
    return X_train, X_test, y_train, y_test
    

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
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass

df = import_data("./data/bank_data.csv")
perform_eda(df)
X, y = encoder_helper(df, ['Gender', 'Education_Level', 'Marital_Status', 
             'Income_Category', 'Card_Category'], response='y')
X_train, X_test, y_train, y_test = perform_feature_engineering(X, y)
print(X_train.info())
print(X_test.info())
print(y_train)
print(y_test)