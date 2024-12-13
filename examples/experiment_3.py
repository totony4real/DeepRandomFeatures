# main.py

import torch
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from DRF.data_utils import get_spherical_data
from DRF.models import initialize_model, DeepMaternRandomPhaseS2RFFNN
from DRF.spherical_uq_methods import SphericalBayesianOptimizer
from DRF.utils import functional_regularisation_S2_batched
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Bayesian Optimization Experiment with ABC datasets"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration YAML file"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"])
    model_name = config["model"]["name"]

    if model_name == "DeepMaternRandomPhaseS2RFFNN":
        file_path = config["data"]["file_path"]
        test_data_path = config["data"]["test_data_path"]
        train_dataset, val_dataset, val_dataset2, test_dataset, d_phi, d_theta = (
            get_spherical_data(file_path, test_data_path, device=device)
        )

        def model_class(*args, **kwargs):
            return DeepMaternRandomPhaseS2RFFNN(*args, **kwargs)

        optimizer = SphericalBayesianOptimizer(
            model_class=DeepMaternRandomPhaseS2RFFNN,
            train_data=train_dataset,
            val_data=val_dataset,
            val_data2=val_dataset2,
            test_data=test_dataset,
            num_layers=config["model"]["num_layers"],
            spatial_input_dim=3,
            temporal_input_dim=1,
            hidden_dim=config["model"]["hidden_dim"],
            bottleneck_dim=config["model"]["bottleneck_dim"],
            output_dim=config["model"]["output_dim"],
            nu=config["model"]["kwargs"].get("nu", 1.5),
            num_models=config["training"]["num_models"],
            d_phi=d_phi,
            d_theta=d_theta,
            device=config["device"],
            p_weight=config["training"]["p_weight"],
            n_iterations=config["bayesian_optimization"]["n_iterations"],
            n_initial_samples=config["bayesian_optimization"]["initial_samples"],
            n_epochs=config["training"]["num_epochs"],
        )

        best_hyperparams, best_loss = optimizer.optimize(
            n_iterations=config["bayesian_optimization"]["n_iterations"]
        )

        (
            spatial_lengthscale,
            temporal_lengthscale,
            amplitude,
            lengthscale2,
            amplitude2,
        ) = best_hyperparams
        print(f"Best hyperparameters:")
        print(f"  spatial_lengthscale = {spatial_lengthscale.item()}")
        print(f"  temporal_lengthscale = {temporal_lengthscale.item()}")
        print(f"  amplitude = {amplitude.item()}")
        print(f"  lengthscale2 = {lengthscale2.item()}")
        print(f"  amplitude2 = {amplitude2.item()}")
        print(f"Best validation loss: {best_loss.item()}")

        top_prediction = sorted(
            optimizer.test_predictions_per_iteration, key=lambda x: x[1]
        )[:1]

        def extract_tensors_and_params(pred_tuple):
            return pred_tuple[0], pred_tuple[2]

        def process_prediction_set(tensor_tuple):
            # Convert each element to a PyTorch tensor if it's not already one
            tensor_tuple = [
                torch.tensor(t) if isinstance(t, np.ndarray) else t
                for t in tensor_tuple
            ]
            return torch.stack(tensor_tuple)

        extracted_predictions_and_params = [
            extract_tensors_and_params(pred) for pred in top_prediction
        ]
        processed_predictions = [
            process_prediction_set(pred[0]) for pred in extracted_predictions_and_params
        ]
        print("Shape of processed predictions:", processed_predictions[0].shape)
        final_test_predictions = processed_predictions[0].mean(dim=0)
        var_final_pred = processed_predictions[0].var(dim=0)

        import pandas as pd

        nll_hyperparams_list = []
        for pred, nll, hyperparams in top_prediction:
            # entry = {'NLL': nll,'Mean huber loss':mean_huber_loss,'CRPS':crps}
            entry = {"NLL": nll}

            if isinstance(hyperparams, dict):
                entry.update(hyperparams)
            elif isinstance(hyperparams, (list, tuple)):
                for i, param in enumerate(hyperparams):
                    entry[f"param_{i}"] = (
                        param.item() if torch.is_tensor(param) else param
                    )
            else:
                entry["param"] = hyperparams
            nll_hyperparams_list.append(entry)

        df = pd.DataFrame(nll_hyperparams_list)
        csv_filename = config["results"]["csv_filename"]
        df.to_csv(csv_filename, index=False)
        print(f"NLL and hyperparameters saved to {csv_filename}")

        torch.save(final_test_predictions, config["results"]["predictions_filename"])
        torch.save(var_final_pred, config["results"]["variance_filename"])
        torch.save(
            processed_predictions[0],
            config["results"]["individual_predictions_filename"],
        )
        _NUM_LONGS = 512
        _NUM_LATS = 256
        print("Final test predictions saved.")
        fig, ax = plt.subplots(
            subplot_kw={"projection": ccrs.PlateCarree(central_longitude=0)}
        )
        ssha_plot = ax.imshow(
            final_test_predictions.reshape(_NUM_LONGS, _NUM_LATS).T,
            origin="lower",
            cmap="coolwarm",
            extent=[0, 360, -90, 90],
            vmin=-0.25,
            vmax=0.25,
        )
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        output_path_nn = config["results"]["plot_mean_filename"]
        plt.savefig(output_path_nn, dpi=300)
        plt.show()

        fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=0)}
        )
        variance_plot = ax.imshow(
            var_final_pred.reshape(_NUM_LONGS, _NUM_LATS).T,
            origin="lower",
            cmap="viridis",  
            extent=[0, 360, -90, 90],
            vmin=0,  
            vmax=0.2,
        )
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        output_path_variance = config["results"]["plot_variance_filename"]
        plt.colorbar(variance_plot, ax=ax, orientation="horizontal", pad=0.05, label="Variance")
        plt.savefig(output_path_variance, dpi=300)
        plt.show()
