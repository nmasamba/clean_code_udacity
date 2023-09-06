# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The project involves writing clean code for churn prediction. At its core is **churn_library.py**, an end-to-end script for predicting customer churn with data import, EDA, feature engineering and model training functions.

To test the main functions in churn_library run: **churn_script_logging_and_tests.py**

## Files and data description
Top level view
---------------
data
- bank_data.csv

churn_library.py
churn_notebook.py
churn_script_logging_and_tests.py
Guide.ipynb

images
- eda
	- churn_hist.png
	- customer_age_hist.png
	- heatmap.png
	- marital_status_bar.png
	- total_transactions_hist.png
- results
	- feat_importances.png
	- linear_reg_clf_results.png
	- random_forest_clf_results.png
	- roc_curve.png

logs
- churn_library.log

models
- logistic_model.pkl
- rfc_model.pkl

README.md
requirements_py3.6.txt
requirements_py3.8.txt

## Running Files
The project was built from within an Anaconda environment. To run it in an isolated virtual environment, make sure you have Continuum's Anaconda installed and follow these steps.

1. Create a new virtual environment with your preferred Python version: 
conda create --name <env_name> python=3.8

2. Activate the new virtual environment:
conda activate <env_name> (to deactivate: conda deactivate)

3. Install pip in the new virtual environment:
conda install pip

4. Use pip to install **dependencies**:
pip install -r requirements_py3.8.txt

5. Open the project files in Jupyter Notebook. For best results, install ipykernel so that the new virtual environment with the required packages can be accessed directly from the notebook:
conda install ipykernel





