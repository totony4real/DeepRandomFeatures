import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
from uq_methods import DeepEnsemble
from data_utils import create_datasets_and_loaders
from utils import calculate_batch_nll


class BayesianOptimiser:
    """
    A class for performing Bayesian optimization using a selected uncertainty quantification (UQ) method.
    
    Attributes:
        config (dict): Configuration dictionary containing hyperparameters for the optimization.
        model_init_func (function): Function to initialize the model.
        device (torch.device): Device to run the optimization on (CPU or GPU).
        bounds (torch.Tensor): Bounds for the hyperparameters used in Bayesian optimization.
        observation_noise_bounds (list): Bounds for observation noise during optimization.
        num_models (int): Number of models for the deep ensemble.
        num_epochs (int): Number of epochs to train each model.
        test_predictions_per_iteration (list): Stores predictions and losses per iteration.
    """
    def __init__(self, config, model_init_func):
        """
        Initializes the BayesianOptimiser with the given configuration and model initialization function.

        Args:
            config (dict): Configuration containing model, training, and optimization settings.
            model_init_func (function): A function to initialize the model for the optimizer.
        """
        self.config = config
        self.device = torch.device(self.config['device'])
        self.bounds = torch.tensor(self.config['bayesian_optimization']['bounds'], dtype=torch.float32).T
        self.observation_noise_bounds = self.config['bayesian_optimization']['observation_noise_bounds']
        self.model_init_func = model_init_func
        self.num_models = self.config['training']['num_models']
        self.num_epochs = self.config['training']['num_epochs']
        self.initialize_uq_method()
        self.test_predictions_per_iteration = []

    def initialize_uq_method(self):
        """
        Initializes the uncertainty quantification method specified in the configuration.
        Currently supports the 'deep_ensemble' method.
        """
        uq_method_type = self.config['uq_method']['type']
        if uq_method_type == 'deep_ensemble':
            self.uq_method = DeepEnsemble(
                model_init_func=self.model_init_func,
                num_models=self.num_models,
                device=self.device,
                num_epochs=self.num_epochs
            )
        else:
            raise ValueError("Unknown UQ method specified.")

    def optimize(self, spatial_X_train, temporal_X_train, y_train, spatial_X_test, temporal_X_test):
        """
        Runs the Bayesian optimization process to find the best hyperparameters by minimizing 
        the negative log-likelihood (NLL) over the training and validation data.

        Args:
            spatial_X_train (torch.Tensor): Spatial training inputs.
            temporal_X_train (torch.Tensor): Temporal training inputs.
            y_train (torch.Tensor): Training target values.
            spatial_X_test (torch.Tensor): Spatial test inputs.
            temporal_X_test (torch.Tensor): Temporal test inputs.

        Returns:
            best_hyperparams (torch.Tensor): The set of hyperparameters that produced the lowest loss.
            best_loss (float): The minimum loss obtained during optimization.
        """
        from pyDOE import lhs
        n_initial_samples = self.config['bayesian_optimization']['initial_samples']
        num_hyperparameters = self.bounds.shape[1]
        lhs_samples = lhs(num_hyperparameters, samples=n_initial_samples)
        lhs_samples = torch.tensor(lhs_samples, dtype=torch.float32)
        train_x = lhs_samples * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        print(train_x)
        train_x = train_x.to(self.device)
        train_y = torch.tensor([
            self.objective_function(
                x,
                spatial_X_train,
                temporal_X_train,
                y_train,
                spatial_X_test,
                temporal_X_test
            ) for x in train_x
        ], dtype=torch.float32).unsqueeze(-1).to(self.device)
        model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        for iteration in range(self.config['bayesian_optimization']['n_iterations']):
            ei = ExpectedImprovement(model, best_f=train_y.min(), maximize=False)
            new_x, _ = optimize_acqf(
                ei,
                bounds=self.bounds.to(self.device),
                q=1,
                num_restarts=100,
                raw_samples=100,
            )
            new_y = torch.tensor([
                self.objective_function(
                    new_x.squeeze(0),
                    spatial_X_train,
                    temporal_X_train,
                    y_train,
                    spatial_X_test,
                    temporal_X_test
                )
            ], dtype=torch.float32).to(self.device)
            train_x = torch.cat([train_x, new_x])
            train_y = torch.cat([train_y, new_y.unsqueeze(0)])
            model = SingleTaskGP(train_x, train_y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)
            print(f"Iteration {iteration + 1}: Spatial Lengthscale = {new_x[0, 0].item()}, Temporal Lengthscale = {new_x[0, 1].item()}, Amplitude = {new_x[0, 2].item()}, Loss = {new_y.item()}")
        
        best_loss, best_idx = train_y.min(0)
        best_hyperparams = train_x[best_idx]

        return best_hyperparams, best_loss.item()

    def objective_function(self, hyperparams, spatial_X_train, temporal_X_train, y_train, spatial_X_test, temporal_X_test):
        """
        Objective function that calculates the negative log-likelihood (NLL) for a given set of hyperparameters.
        It trains the model using the provided spatial and temporal inputs and evaluates its performance on the validation data.

        Args:
            hyperparams (torch.Tensor): Set of hyperparameters to be evaluated.
            spatial_X_train (torch.Tensor): Spatial training inputs.
            temporal_X_train (torch.Tensor): Temporal training inputs.
            y_train (torch.Tensor): Training target values.
            spatial_X_test (torch.Tensor): Spatial test inputs.
            temporal_X_test (torch.Tensor): Temporal test inputs.

        Returns:
            float: The calculated negative log-likelihood (NLL) loss.
        """
        spatial_lengthscale = hyperparams[0].item()
        temporal_lengthscale = hyperparams[1].item()
        amplitude = hyperparams[2].item()
        train_loader, val_loader = create_datasets_and_loaders(
            spatial_X_train, temporal_X_train, y_train,
            batch_size=self.config['training']['batch_size']
        )
        all_predictions, test_predictions = self.uq_method.train_and_evaluate(
            spatial_lengthscale, temporal_lengthscale, amplitude,
            train_loader, val_loader, spatial_X_test, temporal_X_test
        )
        mu = torch.mean(all_predictions, dim=0)
        var = torch.var(all_predictions, dim=0)
        y_val = torch.cat([targets for _, _, targets in val_loader], dim=0).to(self.device)
        nll_loss = calculate_batch_nll(mu, var, y_val, batch_size=self.config['training']['batch_size'])
        self.test_predictions_per_iteration.append((test_predictions, nll_loss, hyperparams))
        return nll_loss
