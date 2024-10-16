import torch
import yaml
from BayesianOptimiser import BayesianOptimiser
from data_utils import load_data

if __name__ == "__main__":
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    spatial_X_train, temporal_X_train, y_train, spatial_X_test, temporal_X_test = load_data()
    optimizer = BayesianOptimiser(config)
    optimizer.optimize(spatial_X_train, temporal_X_train, y_train, spatial_X_test, temporal_X_test)
