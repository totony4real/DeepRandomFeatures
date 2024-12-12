import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
from DRF.uq_methods import DeepEnsemble
from DRF.data_utils import create_datasets_and_loaders
from DRF.utils import calculate_batch_nll


class BayesianOptimiser:
    """
    A class for performing Bayesian optimization using a selected uncertainty quantification (UQ) method.

    Attributes:
        config (dict): Configuration dictionary containing hyperparameters for the optimization.
        model_init_func (function): Function to initialize the model.
        device (torch.device): Device to run the optimization on (CPU or GPU).
        bounds (torch.Tensor): Bounds for the hyperparameters used in Bayesian optimization.
        observation_noise_bounds (list): Bounds for observation noise during optimization (optional).
        num_models (int): Number of models for the deep ensemble.
        num_epochs (int): Number of epochs to train each model.
        test_predictions_per_iteration (list): Stores predictions and losses per iteration.
    """

    def __init__(
        self,
        config,
        model_init_func,
        optimize_observation_noise=False,
        penalty_weight=0,
    ):
        """
        Initializes the BayesianOptimiser with the given configuration and model initialization function.

        Args:
            config (dict): Configuration containing model, training, and optimization settings.
            model_init_func (function): A function to initialize the model for the optimizer.
            optimize_observation_noise (bool): Flag indicating if observation noise should be optimized.
            penalty_weight (float): Strength of the functional regularisation
        """
        self.config = config
        self.device = torch.device(self.config["device"])
        self.bounds = torch.tensor(
            self.config["bayesian_optimization"]["bounds"], dtype=torch.float32
        ).T
        self.optimize_observation_noise = optimize_observation_noise
        self.penalty_weight = penalty_weight
        if self.optimize_observation_noise:
            self.observation_noise_bounds = torch.tensor(
                self.config["bayesian_optimization"]["observation_noise_bounds"]
            ).float()
        self.model_init_func = model_init_func
        self.num_models = self.config["training"]["num_models"]
        self.num_epochs = self.config["training"]["num_epochs"]
        self.initialize_uq_method()
        self.test_predictions_per_iteration = []

    def initialize_uq_method(self):
        uq_method_type = self.config["uq_method"]["type"]
        if uq_method_type == "deep_ensemble":
            self.uq_method = DeepEnsemble(
                model_init_func=self.model_init_func,
                num_models=self.num_models,
                device=self.device,
                num_epochs=self.num_epochs,
            )
        else:
            raise ValueError("Unknown UQ method specified.")

    def optimize(
        self,
        spatial_X_train,
        temporal_X_train,
        y_train,
        spatial_X_test,
        temporal_X_test,
    ):
        """
        Runs the Bayesian optimization process to find the best hyperparameters.

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
        n_initial_samples = self.config["bayesian_optimization"]["initial_samples"]
        num_hyperparameters = self.bounds.shape[1]
        self.bounds = self.bounds.T
        if self.optimize_observation_noise:
            num_hyperparameters += 1
            train_x = (
                torch.rand(n_initial_samples, num_hyperparameters - 1)
                * (self.bounds[:, 1] - self.bounds[:, 0])
                + self.bounds[:, 0]
            )
        else:
            train_x = (
                torch.rand(n_initial_samples, num_hyperparameters)
                * (self.bounds[:, 1] - self.bounds[:, 0])
                + self.bounds[:, 0]
            )

        if self.optimize_observation_noise:
            # obs_noise_samples = torch.rand(n_initial_samples, 1).to(self.device)
            # obs_noise_samples = obs_noise_samples * (self.observation_noise_bounds[1] - self.observation_noise_bounds[0]) + self.observation_noise_bounds[0]
            observation_noise_samples = (
                torch.rand(n_initial_samples, 1)
                * (self.observation_noise_bounds[1] - self.observation_noise_bounds[0])
                + self.observation_noise_bounds[0]
            )
            train_x = torch.cat((train_x, observation_noise_samples), dim=1)

        train_y = (
            torch.tensor(
                [
                    self.objective_function(
                        x,
                        spatial_X_train,
                        temporal_X_train,
                        y_train,
                        spatial_X_test,
                        temporal_X_test,
                    )
                    for x in train_x
                ],
                dtype=torch.float32,
            )
            .unsqueeze(-1)
            .to(self.device)
        )

        model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        for iteration in range(self.config["bayesian_optimization"]["n_iterations"]):
            ei = ExpectedImprovement(model, best_f=train_y.min(), maximize=False)
            if self.optimize_observation_noise:

                print("self.bounds shape:", self.bounds.shape)
                print(
                    "self.observation_noise_bounds shape:",
                    self.observation_noise_bounds.shape,
                )

                obs_noise_bounds = self.observation_noise_bounds.unsqueeze(0)
                print("obs_noise_bounds reshaped:", obs_noise_bounds.shape)

                combined_bounds = torch.cat([self.bounds, obs_noise_bounds]).T
                print("combined_bounds shape:", combined_bounds.shape)
            else:
                combined_bounds = self.bounds.T
            new_x, _ = optimize_acqf(
                ei,
                bounds=combined_bounds,
                q=1,
                num_restarts=10,
                raw_samples=20,
            )
            new_y = torch.tensor(
                [
                    self.objective_function(
                        new_x.squeeze(0),
                        spatial_X_train,
                        temporal_X_train,
                        y_train,
                        spatial_X_test,
                        temporal_X_test,
                    )
                ],
                dtype=torch.float32,
            ).to(self.device)

            train_x = torch.cat([train_x, new_x])
            train_y = torch.cat([train_y, new_y.unsqueeze(0)])
            model = SingleTaskGP(train_x, train_y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            if self.optimize_observation_noise:
                print(
                    f"Iteration {iteration + 1}: Spatial Lengthscale = {new_x[0, 0].item()}, Temporal Lengthscale = {new_x[0, 1].item()}, Amplitude = {new_x[0, 2].item()}, Observation Noise = {new_x[0, 3].item()}, Loss = {new_y.item()}"
                )
            else:
                print(
                    f"Iteration {iteration + 1}: Spatial Lengthscale = {new_x[0, 0].item()}, Temporal Lengthscale = {new_x[0, 1].item()}, Amplitude = {new_x[0, 2].item()}, Loss = {new_y.item()}"
                )

        best_loss, best_idx = train_y.min(0)
        best_hyperparams = train_x[best_idx.to(train_x.device)]

        return best_hyperparams, best_loss.item()

    def objective_function(
        self,
        hyperparams,
        spatial_X_train,
        temporal_X_train,
        y_train,
        spatial_X_test,
        temporal_X_test,
    ):
        """
        Calculates the negative log-likelihood (NLL) for a given set of hyperparameters.

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
        observation_noise = (
            hyperparams[3].item() if self.optimize_observation_noise else 0.00
        )  # default if not optimizing

        train_loader, val_loader = create_datasets_and_loaders(
            spatial_X_train,
            temporal_X_train,
            y_train,
            batch_size=self.config["training"]["batch_size"],
        )
        y_val_list = []
        spatial_val_list = []
        temporal_val_list = []
        for spatial_input, temporal_input, labels in val_loader:
            labels = labels.to(self.device)
            spatial_val_list.append(spatial_input)
            temporal_val_list.append(temporal_input)
            y_val_list.append(labels)
        y_val = torch.cat(y_val_list).to(self.device)
        spatial_val = torch.cat(spatial_val_list).to(self.device)
        temporal_val = torch.cat(temporal_val_list).to(self.device)

        all_predictions, test_predictions, trained_models = (
            self.uq_method.train_and_evaluate(
                spatial_lengthscale,
                temporal_lengthscale,
                amplitude,
                train_loader,
                val_loader,
                spatial_X_test,
                temporal_X_test,
                obs_noise=observation_noise,
            )
        )
        mu = torch.mean(all_predictions, dim=0)
        var = torch.var(all_predictions, dim=0)
        var += observation_noise
        model_instance = trained_models[0]
        y_val = torch.cat([targets for _, _, targets in val_loader], dim=0).to(
            self.device
        )
        nll_loss, only_nll_val = calculate_batch_nll(
            mu,
            var,
            y_val,
            spatial_val,
            temporal_val,
            model=model_instance,
            batch_size=self.config["training"]["batch_size"],
            gradient_penalty_weight=self.penalty_weight,
        )

        self.test_predictions_per_iteration.append(
            (test_predictions, nll_loss, hyperparams)
        )

        return nll_loss
