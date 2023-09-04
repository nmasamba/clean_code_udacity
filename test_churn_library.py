

'''
Author: Nyasha Masamba
Date: 3rd September 2023

Purpose of program
-------------------
Example pytest script for testing the import_data function in churn_library.py.
Complete code for testing churn_library can be found in churn_script_logging_and_tests.py (not using pytest).
'''

from churn_library import import_data

def test_import_data_shape():
    try:
        assert(import_data("./data/bank_data.csv").shape[0] > 0)
        assert(import_data("./data/bank_data.csv").shape[1] > 0)
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err

def test_import_data_is_df():
    import pandas as pd
    assert(isinstance(import_data("./data/bank_data.csv"), pd.DataFrame))