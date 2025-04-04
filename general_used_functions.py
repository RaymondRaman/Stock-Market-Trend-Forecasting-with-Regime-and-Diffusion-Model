
import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
import joblib


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


def load_training_data():
    config_data = load_config_file()

    # Load training data
    DATA_DIR = os.getcwd() + '/data'
    training_stock_df = defaultdict(list)
    stock_list = config_data['stock_dict'].keys()

    for stock in stock_list:
        feature_data = pd.read_excel(
            f'{DATA_DIR}/training_data_full/training_feature_{stock}.xlsx')
        stock_data = pd.read_excel(
            f'{DATA_DIR}/stock_price_data/training/{stock}_stock_price_data(training).xlsx')

        # Merge the two dataframes on 'date'
        merged_data = pd.merge(feature_data, stock_data, on='date', how='left')

        # # Drop the 'date' column from the feature data
        # merged_data.drop(columns=['date'], inplace=True)

        # Store the merged data in the dictionary
        training_stock_df[stock] = merged_data

    return training_stock_df


def load_testing_data():
    config_data = load_config_file()

    # Load testing data
    DATA_DIR = os.getcwd() + '/data'
    testing_stock_df = defaultdict(list)
    stock_list = config_data['stock_dict'].keys()

    for stock in stock_list:
        feature_data = pd.read_excel(
            f'{DATA_DIR}/testing_data_full/testing_feature_{stock}.xlsx')
        stock_data = pd.read_excel(
            f'{DATA_DIR}/stock_price_data/testing/{stock}_stock_price_data(testing).xlsx')

        # Merge the two dataframes on 'date'
        merged_data = pd.merge(feature_data, stock_data, on='date', how='left')

        # # Drop the 'date' column from the feature data
        # merged_data.drop(columns=['date'], inplace=True)

        # Store the merged data in the dictionary
        testing_stock_df[stock] = merged_data

    return testing_stock_df


# Define a fixed palette mapping for up to 10 states
FIXED_COLORS = {
    0: "blue",
    1: "green",
    2: "red",
    3: "purple",
    4: "orange",
    5: "cyan",
    6: "magenta",
    7: "brown",
    8: "pink",
    9: "grey"
}


def plot_market_regime(states, stock, is_test=False, fixed_palette=FIXED_COLORS, isHMM=True):
    sns.set(font_scale=15)
    # Determine the unique states present in the current dataset
    unique_states = sorted(states['states'].unique())
    # Create a consistent color mapping based on the fixed_palette.
    custom_palette = {state: fixed_palette[state] for state in unique_states}

    sns.set(style="whitegrid")
    fg = sns.FacetGrid(
        data=states,
        hue='states',
        hue_order=unique_states,
        palette=custom_palette,
        aspect=1.31,
        legend_out=False
    )
    subscripts = "(Training)" if not is_test else "(Testing)"

    fg.map(plt.scatter, 'Date', stock, alpha=0.8)
    fg.add_legend(prop={'size': 20}, title="States", title_fontsize='20')

    if not is_test:
        fg.fig.suptitle(f'Historical {stock} Market Trend {subscripts}',
                        fontsize=80, fontweight='bold')
        fg.fig.set_size_inches(60, 30)
    else:
        # Smaller size for testing
        fg.fig.set_size_inches(10, 10)
        fg.fig.suptitle(f'Historical {stock} Market Trend {subscripts}',
                        fontsize=20, fontweight='bold')
    sns.despine(offset=10)

    # Save the plot in HMM_trend_detection directory
    DATA_DIR = os.getcwd() + \
        f'/trend_detection/{stock}' if isHMM else os.getcwd() + \
        f'/trend_detection/{stock}'
    if is_test:
        save_path = f"{DATA_DIR}/{stock}_HMM_test_trend_detection.png" if isHMM else f"{DATA_DIR}/{stock}_SJM_test_trend_detection.png"
        plt.savefig(save_path, dpi=300)
    else:
        os.makedirs(DATA_DIR, exist_ok=True)
        save_path = f"{DATA_DIR}/{stock}_HMM_train_trend_detection.png" if isHMM else f"{DATA_DIR}/{stock}_SJM_train_trend_detection.png"
        plt.savefig(save_path, dpi=300)

    plt.show()


def HMM_training(stock, train_df, n_components=3, n_iter=1000000):
    # Exclude the date column when training the model
    features = train_df.drop(columns=['date'])
    model = hmm.GaussianHMM(n_components=n_components,
                            covariance_type="full", n_iter=n_iter)
    model.fit(features)
    hidden_states = model.predict(features)

    # Build the states DataFrame with Date, the stock price column, and hidden states.
    states = pd.DataFrame({
        'Date': train_df['date'],
        stock: train_df[stock],
        'states': hidden_states
    })

    plot_market_regime(states, stock)
    return model, states


def HMM_testing(stock, test_df, model):
    # Exclude the date column when testing the model
    features = test_df.drop(columns=['date'])
    hidden_states = model.predict(features)

    # Build the states DataFrame with Date, the stock price column, and hidden states.
    states = pd.DataFrame({
        'Date': test_df['date'],
        stock: test_df[stock],
        'states': hidden_states
    })

    plot_market_regime(states, stock, True)

    return states


def save_HMM_states_excel(stock, states, is_test=False):
    # Load training data
    DATA_DIR = os.getcwd() + '/data'
    subscript = "(Testing)" if is_test else "(Training)"

    # If the directory does not exist, create it
    os.makedirs(f"{DATA_DIR}/HMM_states/{stock}", exist_ok=True)

    states.to_excel(
        f"{DATA_DIR}/HMM_states/{stock}/{stock}_HMM_states{subscript}.xlsx", index=False)


def save_HMM_model(stock, model):
    # Define the directory for saving models and create it if it doesn't exist
    DATA_DIR = os.path.join(os.getcwd(), 'model')
    os.makedirs(DATA_DIR, exist_ok=True)

    # Save the model using joblib
    model_filename = os.path.join(DATA_DIR, f'{stock}_HMM_model.joblib')
    joblib.dump(model, model_filename)
