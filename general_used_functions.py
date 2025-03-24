
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


# # Training
# for stock, weird_columns in training_weird_columns_dict.items():
#     if weird_columns:
#         training_feature_df[stock] = handle_weird_data_with_knn_imputer(training_feature_df[stock], weird_columns)

# # Testing
# for stock, weird_columns in testing_weird_columns_dict.items():
#     if weird_columns:
#         testing_feature_df[stock] = handle_weird_data_with_knn_imputer(testing_feature_df[stock], weird_columns)

# # Check the data again
# for stock, stock_data in training_feature_df.items():
#     if check_weird_data(stock_data):
#         print(f"The {stock} stock price data still contains weird data in training data")

# for stock, stock_data in testing_feature_df.items():
#     if check_weird_data(stock_data):
#         print(f"The {stock} stock price data still contains weird data in testing data")

# # Clear weird_columns_dict, only clear the value, not the key
# for stock in stock_list:
#     training_weird_columns_dict[stock] = []

# for stock in stock_list:
#     testing_weird_columns_dict[stock] = []

# # Print the result to ensure the data is correct
# print(training_feature_df['AAPL'].head().to_string())
# print(testing_feature_df['AAPL'].head().to_string())
