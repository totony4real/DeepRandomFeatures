import torch
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class DeepEnsemble:
    """
    A class for training and evaluating an ensemble of deep learning models in parallel.
    
    Attributes:
        model_init_func (function): Function to initialize each model in the ensemble.
        num_models (int): Number of models in the ensemble.
        device (torch.device): Device on which models will be trained (CPU or GPU).
        num_epochs (int): Number of training epochs for each model.
    """
    def __init__(self, model_init_func, num_models, device, num_epochs):
        """
        Initializes the DeepEnsemble class.

        Args:
            model_init_func (function): A function to initialize the models.
            num_models (int): Number of models to train in the ensemble.
            device (torch.device): The device (CPU or GPU) to run the training on.
            num_epochs (int): The number of epochs to train each model.
        """
        self.model_init_func = model_init_func
        self.num_models = num_models
        self.device = device
        self.num_epochs = num_epochs
    
    def map_loss(self,outputs, targets, model_parameters, s_squared):
        """
        Computes the Maximum A Posteriori (MAP) loss, which is a combination of the mean squared error (MSE) and 
        a regularization term on the model parameters (L2 regularization).

        Args:
            outputs (torch.Tensor): The predicted outputs from the model.
            targets (torch.Tensor): The ground truth target values.
            model_parameters (iterable): An iterable (e.g., a list) of the model's parameters (weights).
            s_squared (float): A scaling factor for the regularization term (variance term in MAP).

        Returns:
            torch.Tensor: The total MAP loss, which is the sum of the MSE and the regularization term.
        """
        mse_term = ((outputs - targets) ** 2).sum() / (2*1e-2**2)
        reg_term = 0.5* sum(param.pow(2).sum() for param in model_parameters)  / s_squared

        total_loss = mse_term + reg_term
        return total_loss

    def train_and_evaluate(self, spatial_lengthscale, temporal_lengthscale, amplitude, train_loader, val_loader, spatial_X_test, temporal_X_test):
        """
        Trains the ensemble models in parallel and evaluates them on the test data.

        Args:
            spatial_lengthscale (float): Lengthscale for the spatial input.
            temporal_lengthscale (float): Lengthscale for the temporal input.
            amplitude (float): Amplitude for the model.
            train_loader (DataLoader): DataLoader for the training data.
            val_loader (DataLoader): DataLoader for the validation data.
            spatial_X_test (torch.Tensor): Spatial test inputs.
            temporal_X_test (torch.Tensor): Temporal test inputs.

        Returns:
            tuple: A tuple containing stacked predictions from all models and individual test predictions.
        """
        results = Parallel(n_jobs=self.num_models)(
            delayed(self._train_single_model)(
                spatial_lengthscale, temporal_lengthscale, amplitude,
                train_loader, val_loader, spatial_X_test, temporal_X_test, model_index
            ) for model_index in range(self.num_models)
        )
        all_predictions, test_predictions = zip(*results)
        return torch.stack(all_predictions).squeeze().to(self.device), test_predictions

    def _train_single_model(self, spatial_lengthscale, temporal_lengthscale, amplitude, train_loader, val_loader, spatial_X_test, temporal_X_test, model_index):
        """
        Trains a single model in the ensemble and evaluates it on validation and test data.

        Args:
            spatial_lengthscale (float): Lengthscale for the spatial input.
            temporal_lengthscale (float): Lengthscale for the temporal input.
            amplitude (float): Amplitude for the model.
            train_loader (DataLoader): DataLoader for the training data.
            val_loader (DataLoader): DataLoader for the validation data.
            spatial_X_test (torch.Tensor): Spatial test inputs.
            temporal_X_test (torch.Tensor): Temporal test inputs.
            model_index (int): Index of the model in the ensemble.

        Returns:
            tuple: A tuple containing validation predictions and test predictions for the model.
        """
        model = self.model_init_func(spatial_lengthscale, temporal_lengthscale, amplitude)
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        model.train()
        for epoch in range(self.num_epochs):
            train_loader_tqdm = tqdm(
                train_loader,
                desc=f"Model {model_index+1}/{self.num_models} - Epoch {epoch+1}/{self.num_epochs}",
                leave=False
            )
            for spatial_input, temporal_input, targets in train_loader_tqdm:
                spatial_input = spatial_input.to(self.device)
                temporal_input = temporal_input.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                s_squared = amplitude**2
                outputs = model(spatial_input, temporal_input)
                loss = self.map_loss(outputs, targets.unsqueeze(1), model.parameters(), s_squared)
                loss.backward()
                optimizer.step()
                train_loader_tqdm.set_postfix(loss=loss.item())

        model.eval()
        predictions = []
        with torch.no_grad():
            val_loader_tqdm = tqdm(
                val_loader,
                desc=f"Model {model_index+1}/{self.num_models} - Validation",
                leave=False
            )
            for spatial_input, temporal_input, targets in val_loader_tqdm:
                spatial_input = spatial_input.to(self.device)
                temporal_input = temporal_input.to(self.device)
                outputs = model(spatial_input, temporal_input)
                predictions.append(outputs.cpu())

        final_preds = []
        test_dataset = TensorDataset(spatial_X_test, temporal_X_test)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
        with torch.no_grad():
            test_loader_tqdm = tqdm(
                test_loader,
                desc=f"Model {model_index+1}/{self.num_models} - Testing",
                leave=False
            )
            for spatial_batch, temporal_batch in test_loader_tqdm:
                spatial_batch = spatial_batch.to(self.device)
                temporal_batch = temporal_batch.to(self.device)
                preds = model(spatial_batch, temporal_batch).view(-1, 1)
                final_preds.append(preds.cpu())

        final_preds = torch.cat(final_preds, dim=0)
        return torch.cat(predictions).unsqueeze(0), final_preds

