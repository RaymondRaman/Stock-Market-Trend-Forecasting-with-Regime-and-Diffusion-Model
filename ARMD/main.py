import warnings
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Models.autoregressive_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, instantiate_from_config
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import get_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from engine.solver import Trainer
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import random
import argparse
import sys

# Append the project's root directory (i.e., one level up from the ARMD folder) to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

warnings.filterwarnings("ignore")


def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Parameters:
    - seed (int): The seed value.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Additional steps for CuDNN backend
    os.environ['PYTHONHASHSEED'] = str(seed)


# Example usage:
set_seed(2023)

# class Args_Example:
#    def __init__(self) -> None:
#        self.config_path = './Config/etth.yaml'
#        self.save_dir = './forecasting_exp'
#        self.gpu = 0
#        os.makedirs(self.save_dir, exist_ok=True)


class Args_Example:
    def __init__(self, config_path, save_dir, gpu):
        self.config_path = config_path
        self.save_dir = save_dir
        self.gpu = gpu
        os.makedirs(self.save_dir, exist_ok=True)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process configuration and directories.")
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the configuration file.')
    parser.add_argument('--save_dir', type=str, default='./forecasting_exp',
                        help='Directory to save experiment results.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Specify which GPU to use.')
    parser.add_argument("--state", type=str, default="false",
                        help="Signal for using state features (true/false)")
    parser.add_argument("--stock_ticker", type=str,
                        help="Stock ticker symbol for the dataset")

    args = parser.parse_args()
    return args


def plot_forcasting(sampled_forecast_values, real_future_values, ticker, context_len, pred_len, historical_context, last_historical_point):
    # Line 1: Full Actual Series (Historical Context + Real Future Values)
    # X-axis for the full actual series
    x_full_actual = np.arange(0, context_len + pred_len)
    # Y-axis for the full actual series
    y_full_actual = np.concatenate((historical_context, real_future_values))

    # Line 2: Sampled Forecast (connected to the last historical point)
    # Y-axis for sampled forecast, starting with the last historical point
    y_sampled_forecast_connected = np.concatenate(
        ([last_historical_point], sampled_forecast_values))
    # X-axis for this connected forecast line (starts one step before context_len to include last_historical_point)
    x_sampled_forecast_connected = np.arange(
        context_len - 1, context_len - 1 + len(y_sampled_forecast_connected))

    plt.figure(figsize=(12, 6))
    plt.plot(x_full_actual, y_full_actual,
             label='Actual Values (History + Future)', color='green')
    plt.plot(x_sampled_forecast_connected, y_sampled_forecast_connected,
             label='Sampled Forecast (Connected to History)', color='blue', linestyle='--')

    plt.title(
        f'Real vs. Forecasted Stock Price for {ticker}')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.axvline(x=context_len - 1, color='r', linestyle=':', linewidth=1,
                label='Forecast Start')  # Mark where forecast begins
    plt.legend()
    plt.grid(True)

    os.makedirs(f'./final_results/{ticker}', exist_ok=True)
    plt.savefig(
        f'./final_results/{ticker}/{ticker}_forecasting.png')


if __name__ == "__main__":
    # args =  Args_Example()
    args_parsed = parse_arguments()
    args = Args_Example(args_parsed.config_path,
                        args_parsed.save_dir, args_parsed.gpu)

    args.use_state = (args_parsed.state.lower() == "true")

    seq_len = 96
    configs = load_yaml_config(args.config_path)
    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = instantiate_from_config(configs['model']).to(device)
    # model.use_ff = False
    model.fast_sampling = True
    # configs['solver']['max_epochs']=100
    dataloader_info = build_dataloader(configs, args)
    dataloader = dataloader_info['dataloader']
    trainer = Trainer(config=configs, args=args, model=model,
                      dataloader={'dataloader': dataloader})
    trainer.train()

    args.mode = 'predict'
    args.pred_len = seq_len
    test_dataloader_info = build_dataloader_cond(configs, args)
    test_scaled = test_dataloader_info['dataset'].samples
    scaler = test_dataloader_info['dataset'].scaler
    seq_length, feat_num = seq_len*2, test_scaled.shape[-1]
    pred_length = seq_len
    real = test_scaled
    test_dataset = test_dataloader_info['dataset']
    test_dataloader = test_dataloader_info['dataloader']
    sample, real_ = trainer.sample_forecast(
        test_dataloader, shape=[seq_len, feat_num])
    mask = test_dataset.masking

    # Extract the historical context for the first sample, first feature which is the stock price
    context_len = seq_len
    pred_len = seq_len
    historical_context = test_scaled[0, :context_len, 0]
    last_historical_point = historical_context[-1]

    # Forecasted values and actual future values
    sampled_forecast_values, real_future_values = sample[0, :, 0], real_[
        0, :, 0]

    # Compute MSE and MAE for the stock price only
    mse = mean_squared_error(
        sampled_forecast_values.reshape(-1), real_future_values.reshape(-1))
    mae = mean_absolute_error(
        sampled_forecast_values.reshape(-1), real_future_values.reshape(-1))

    # Plotting the results only with the state features
    plot_forcasting(sampled_forecast_values, real_future_values, args_parsed.stock_ticker,
                    context_len, pred_len, historical_context, last_historical_point)
    print(f"MAE: {mae}, MSE: {mse}")
