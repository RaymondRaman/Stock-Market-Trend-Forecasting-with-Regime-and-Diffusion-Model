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


def plot_forcasting(price_forecast_normalized, price_real_normalized, scaler, ticker, substring):
    scale_price = scaler.scale_[0]
    mean_price = scaler.mean_[0]

    price_forecast_original = unnormalize_to_zero_to_one(
        price_forecast_normalized)
    price_forecast_original = price_forecast_original.reshape(
        price_forecast_normalized.shape)

    price_real_original = unnormalize_to_zero_to_one(price_real_normalized)
    price_real_original = price_real_original.reshape(
        price_real_normalized.shape)

    forecast_1_original = price_forecast_normalized[0]
    real_1_original = price_real_normalized[0]
    time_axis = range(len(forecast_1_original))

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, real_1_original, label='Real Stock Price', marker='o')
    plt.plot(time_axis, forecast_1_original,
             label='Forecasted Stock Price', marker='x')
    plt.xlabel('Time Step')
    plt.ylabel('Stock Price')
    plt.title(
        f'Real vs. Forecasted Stock Price for {ticker} {substring}')
    plt.legend()
    os.makedirs(f'./final_results/{ticker}', exist_ok=True)
    plt.savefig(
        f'./final_results/{ticker}/{ticker}_forecasting_{substring}.png')


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

    # Extract only column 0 (the stock price) from both sample and real_
    price_forecast = sample[..., 0]  # shape (n_samples, seq_length)
    price_real = real_[..., 0]        # shape (n_samples, seq_length)

    # Compute MSE and MAE for the stock price only
    mse = mean_squared_error(
        price_forecast.reshape(-1), price_real.reshape(-1))
    mae = mean_absolute_error(
        price_forecast.reshape(-1), price_real.reshape(-1))

    substring = "(including states data)" if args.use_state else "(not including states data)"
    print(f"Results for {substring} features:")
    print("MSE (Stock Price):", mse)
    print("MAE (Stock Price):", mae)

    # Plotting the results only with the state features
    plot_forcasting(price_forecast, price_real, scaler,
                    ticker=args_parsed.stock_ticker, substring=substring)
