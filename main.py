"""
Main execution module for the multi-agent dropout prediction system
"""
import argparse
import os
from runner import runner

def main():
    
    # ───────── CLI + logging
    cli = argparse.ArgumentParser()
    cli.add_argument("--llama-size", choices=["7b","13b"], default="7b")
    cli.add_argument("--time-window", type=int, default=30, help="Days for time series window")
    cli.add_argument("--prediction-horizon", type=int, default=7, help="Days ahead to predict")
    cli.add_argument("--tsdata", default=False, help="Time series data file that ")
    cli.add_argument("--ollama", default=False, help="Default model")
    cli.add_argument("--train_log", default=False, help="save for finetune model")
    cli.add_argument("--N_students",type=int, default=50, help="Number of student")
    cli.add_argument("--retune", action="store_true", help="Force model retuning, ignoring saved models.")
    cli.add_argument("--xuetangx", default=False,
                     help="use XuetangX MOOC-data instead of OULAD")
    # +++ Added for UCI dataset
    cli.add_argument("--uci", default=False, help="Use the UCI Student Dropout dataset.")
    cli.add_argument("--uci_path", type=str, default=os.path.join("raw_datasets", "UCI", "data.csv"), help="Path to the UCI dataset CSV file.")

    # Add these new arguments for XuetangX
    cli.add_argument("--train_path", type=str, default=os.path.join("raw_datasets", "xuetangx", "Train.csv"), help="Path to XuetangX Train.csv")
    cli.add_argument("--test_path", type=str, default=os.path.join("raw_datasets", "xuetangx", "Test.csv"), help="Path to XuetangX Test.csv")
    cli.add_argument("--user_info_path", type=str, default=os.path.join("raw_datasets", "xuetangx", "user_info.csv"), help="Path to XuetangX user_info.csv")
    
    # Add argument for OULAD folder path
    cli.add_argument("--folder_path", type=str, default=os.path.join("raw_datasets", "OULAD", "data"), help="Path to OULAD dataset folder")

    args = cli.parse_args()

    runner(args)
if __name__ == "__main__":
    main()