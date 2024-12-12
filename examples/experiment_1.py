import torch
import yaml
import argparse
import pandas as pd
from DRF.BayesianOptimiser import BayesianOptimiser
from DRF.data_utils import get_mss_data
from DRF.models import initialize_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bayesian Optimization Experiment")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration YAML file"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    zip_path = config["data"]["zip_path"]
    data_folder = config["data"]["data_folder"]
    spatial_X_train, temporal_X_train, y_train, spatial_X_test, temporal_X_test = (
        get_mss_data(zip_path, data_folder)
    )

    device = torch.device(config["device"])

    def model_init_func(spatial_lengthscale, temporal_lengthscale, amplitude, device=device):
        return initialize_model(
            model_name=config['model']['name'],
            num_layers=config['model']['num_layers'],
            spatial_input_dim=spatial_X_train.shape[-1],
            temporal_input_dim=temporal_X_train.shape[-1],
            hidden_dim=config['model']['hidden_dim'],
            bottleneck_dim=config['model']['bottleneck_dim'],
            output_dim=config['model']['output_dim'],
            spatial_lengthscale=spatial_lengthscale,
            temporal_lengthscale=temporal_lengthscale,
            amplitude=amplitude,
            device=device,
            spatial_layer_type=config['model']['spatial_layer_type'],
            temporal_layer_type=config['model']['temporal_layer_type'],
            model_kwargs=config['model'].get('kwargs', {})
        )

    optimizer = BayesianOptimiser(config, model_init_func)

    best_hyperparams, best_loss = optimizer.optimize(
        spatial_X_train, temporal_X_train, y_train, spatial_X_test, temporal_X_test
    )

    print(f"Best Hyperparameters: {best_hyperparams.cpu().numpy()}, Loss: {best_loss}")

    top_prediction = sorted(optimizer.test_predictions_per_iteration, key=lambda x: x[1])[:1]

    def extract_tensors_and_params(pred_tuple):
        return pred_tuple[0], pred_tuple[2]

    def process_prediction_set(tensor_tuple):
        return torch.stack(tensor_tuple)

    extracted_predictions_and_params = [extract_tensors_and_params(pred) for pred in top_prediction]
    processed_predictions = [process_prediction_set(pred[0]) for pred in extracted_predictions_and_params]
    print('Shape of processed predictions:', processed_predictions[0].shape)
    final_test_predictions = processed_predictions[0].mean(dim=0)
    var_final_pred = processed_predictions[0].var(dim=0)

    nll_hyperparams_list = []
    for pred, nll, hyperparams in top_prediction:
        entry = {'NLL': nll}
        if isinstance(hyperparams, dict):
            entry.update(hyperparams)
        elif isinstance(hyperparams, (list, tuple)):
            for i, param in enumerate(hyperparams):
                entry[f'param_{i}'] = param.item() if torch.is_tensor(param) else param
        else:
            entry['param'] = hyperparams
        nll_hyperparams_list.append(entry)

    df = pd.DataFrame(nll_hyperparams_list)
    csv_filename = config['results']['csv_filename']
    df.to_csv(csv_filename, index=False)
    print(f"NLL and hyperparameters saved to {csv_filename}")

    torch.save(final_test_predictions, config['results']['predictions_filename'])
    torch.save(var_final_pred, config['results']['variance_filename'])
    print("Final test predictions saved.")