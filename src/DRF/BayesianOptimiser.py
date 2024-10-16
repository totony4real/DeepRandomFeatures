import torch
import yaml
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
from uq_methods import DeepEnsemble, MCDropout
from data_utils import create_datasets_and_loaders
from DRFSat.models import DeepSpatiotemporalGPNN

class BayesianOptimiser:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config['device'])
        self.bounds = torch.tensor(self.config['bayesian_optimization']['bounds']).T
        self.observation_noise_bounds = self.config['bayesian_optimization']['observation_noise_bounds']
        self.initialize_uq_method()
        self.test_predictions_per_iteration = []

    def initialize_uq_method(self):
        uq_method_type = self.config['uq_method']['type']
        if uq_method_type == 'deep_ensemble':
            self.uq_method = DeepEnsemble(num_models=self.config['training']['num_models'])
        elif uq_method_type == 'mc_dropout':
            self.uq_method = MCDropout(mc_iterations=self.config['uq_method']['mc_iterations'])
        else:
            raise ValueError("Unknown UQ method specified.")

    def optimize(self, spatial_X_train, temporal_X_train, y_train, spatial_X_test, temporal_X_test):
        
        from pyDOE import lhs
        n_initial_samples = self.config['bayesian_optimization']['initial_samples']
        lhs_samples = lhs(3, samples=n_initial_samples)  # 3 hyperparameters here
        train_x = torch.tensor(lhs_samples * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0], dtype=torch.float32)

        observation_noise_samples = torch.rand(n_initial_samples) * (self.observation_noise_bounds[1] - self.observation_noise_bounds[0]) + self.observation_noise_bounds[0]
        train_x = torch.cat([train_x, observation_noise_samples.unsqueeze(-1)], dim=1)
        train_y = torch.tensor([self.objective_function(x, spatial_X_train, temporal_X_train, y_train, spatial_X_test, temporal_X_test) for x in train_x], dtype=torch.float32).unsqueeze(-1)

        model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        for iteration in range(self.config['bayesian_optimization']['n_iterations']):
            ei = ExpectedImprovement(model, best_f=train_y.min(), maximize=False)
            new_x, _ = optimize_acqf(ei, bounds=torch.cat([self.bounds, torch.tensor(self.observation_noise_bounds).unsqueeze(0)]).T, q=1, num_restarts=10, raw_samples=20)
            new_y = torch.tensor([self.objective_function(new_x.squeeze(0), spatial_X_train, temporal_X_train, y_train, spatial_X_test, temporal_X_test)], dtype=torch.float32)
            train_x = torch.cat([train_x, new_x])
            train_y = torch.cat([train_y, new_y.unsqueeze(0)])
            model = SingleTaskGP(train_x, train_y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

    def objective_function(self, hyperparams, spatial_X_train, temporal_X_train, y_train, spatial_X_test, temporal_X_test):
        spatial_lengthscale, temporal_lengthscale, amplitude, observation_noise = hyperparams
        model = DeepSpatiotemporalGPNN(
            num_layers=self.config['model']['num_layers'],
            spatial_input_dim=spatial_X_train.shape[-1],
            temporal_input_dim=temporal_X_train.shape[-1],
            hidden_dim=self.config['model']['hidden_dim'],
            bottleneck_dim=self.config['model']['bottleneck_dim'],
            output_dim=self.config['model']['output_dim'],
            spatial_rff_layer_type=self.config['model']['spatial_layer_type'],
            temporal_rff_layer_type=self.config['model']['temporal_layer_type'],
            spatial_lengthscale=spatial_lengthscale.item(),
            temporal_lengthscale=temporal_lengthscale.item(),
            amplitude=amplitude.item(),
            device=self.device
        ).to(self.device)

        train_loader, val_loader = create_datasets_and_loaders(spatial_X_train, temporal_X_train, y_train, batch_size=self.config['training']['batch_size'])
        return self.uq_method.train_and_evaluate(model, train_loader, val_loader, spatial_X_test, temporal_X_test)
