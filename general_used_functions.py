
import os
import json
import numpy as np
import pandas as pd


def load_config_file():
    # Open the config file
    config_file_path = os.path.join(os.getcwd(), 'config.json')
    with open(config_file_path) as f:
        config = json.load(f)

    return config


def check_weird_data(data):
    """
    Check if data contains NaN, inf, or -inf
    :param data: DataFrame
    """
    if data.isnull().values.any() or data.isin([np.inf, -np.inf]).values.any():
        return True
    return False


# Handle unmatched date format
def parse_dates(date_series):
    return pd.to_datetime(date_series, dayfirst=False)
