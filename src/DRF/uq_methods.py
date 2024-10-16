import torch
from joblib import Parallel, delayed
import torch.nn.functional as F

class DeepEnsemble:
    def __init__(self, model_init_func, num_models, device):
        self.model_init_func = model_init_func
        self.num_models = num_models
        self.device = device

    def train_and_evaluate(self, spatial_lengthscale, temporal_lengthscale, amplitude, train_loader, val_loader):
        results = Parallel(n_jobs=self.num_models)(delayed(self._train_single_model)(
            spatial_lengthscale, temporal_lengthscale, amplitude, train_loader, val_loader) for _ in range(self.num_models)
        )
        return self._aggregate_results(results)

    def _train_single_model(self, spatial_lengthscale, temporal_lengthscale, amplitude, train_loader, val_loader):
        pass

    def _aggregate_results(self, results):
        all_predictions = torch.stack([result[0] for result in results], dim=0)
        mean_pred = all_predictions.mean(dim=0)
        var_pred = all_predictions.var(dim=0)
        return mean_pred, var_pred
