import os
import zipfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

def unzip_data(source_zip, target_folder):
    """
    Unzips the provided source zip file into the target folder. If the folder already exists, it skips the extraction.

    Args:
        source_zip (str): Path to the source zip file.
        target_folder (str): Path to the target folder where data should be extracted.
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)
        with zipfile.ZipFile(source_zip, 'r') as zip_ref:
            zip_ref.extractall(target_folder)
        print("Data unzipped.")
    else:
        print("Data already unzipped.")

def load_data(data_folder):
    """
    Loads the ground truth data, observation data, and test locations from the specified data folder.

    Args:
        data_folder (str): Path to the folder containing the data files.

    Returns:
        tuple: A tuple containing the ground truth DataFrame, observation data DataFrame, and test locations DataFrame.

    Raises:
        FileNotFoundError: If the expected data files are not found in the folder.
    """
    expected_file = os.path.join(data_folder, 'mss_data', 'CryosatMSS-arco-2yr-140821_with_geoid_h.csv')
    if not os.path.exists(expected_file):
        raise FileNotFoundError("Data files not found, please check the data folder or unzip the data first.")
    
    gt_df = pd.read_csv(expected_file)
    hdf5_file_path = os.path.join(data_folder, 'mss_data', 'along_track_sample_from_mss_ground_ABC.h5')
    obs_data = pd.read_hdf(hdf5_file_path, 'data')
    test_locs = pd.read_csv(os.path.join(data_folder, 'mss_data', 'test_locs.csv'))
    
    return gt_df, obs_data, test_locs

def prepare_tensor_datasets(obs_data, test_locs):
    """
    Prepares and normalizes the tensor datasets from the observation data and test locations for training and testing.

    Args:
        obs_data (pd.DataFrame): Observation data containing spatial, temporal, and observation values.
        test_locs (pd.DataFrame): Test locations containing spatial and temporal values.

    Returns:
        tuple: A tuple containing tensors for spatial training inputs, temporal training inputs, training target values,
               spatial test inputs, and temporal test inputs.
    """
    test_locs = add_grid_indices_and_date(test_locs)
    
    spatial_X = obs_data[['x', 'y']].to_numpy()
    temporal_X = obs_data[['t']].to_numpy()
    Y = obs_data['obs'].to_numpy()

    spatial_tensor_x = torch.Tensor(spatial_X)
    temporal_tensor_x = torch.Tensor(temporal_X)
    tensor_y = torch.Tensor(Y)
    spatial_X_mean = spatial_tensor_x.mean(dim=0)
    spatial_X_std = spatial_tensor_x.std(dim=0)
    normalized_spatial_tensor_x = (spatial_tensor_x - spatial_X_mean) / spatial_X_std

    temporal_X_mean = temporal_tensor_x.mean(dim=0)
    temporal_X_std = temporal_tensor_x.std(dim=0)
    normalized_temporal_tensor_x = (temporal_tensor_x - temporal_X_mean) / temporal_X_std

    Y_mean = tensor_y.mean()
    Y_std = tensor_y.std()
    normalized_tensor_y = (tensor_y - Y_mean) / Y_std

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    spatial_X_train_torch = normalized_spatial_tensor_x.to(device)
    temporal_X_train_torch = normalized_temporal_tensor_x.to(device)
    y_train_torch = normalized_tensor_y.to(device)
    spatial_test_X = test_locs[['x', 'y']].to_numpy()
    temporal_test_X = test_locs[['t']].to_numpy()
    normalized_spatial_test_X = (torch.Tensor(spatial_test_X) - spatial_X_mean) / spatial_X_std
    normalized_temporal_test_X = (torch.Tensor(temporal_test_X) - temporal_X_mean) / temporal_X_std

    spatial_X_test_torch = normalized_spatial_test_X.to(device)
    temporal_X_test_torch = normalized_temporal_test_X.to(device)

    return spatial_X_train_torch, temporal_X_train_torch, y_train_torch, spatial_X_test_torch, temporal_X_test_torch

def get_mss_data(zip_path, data_folder):
    """
    Unzips the data, loads it, and prepares the tensor datasets for training and testing.

    Args:
        zip_path (str): Path to the zip file containing the data.
        data_folder (str): Path to the folder where data will be unzipped and processed.

    Returns:
        tuple: A tuple containing tensors for spatial training inputs, temporal training inputs, training target values,
               spatial test inputs, and temporal test inputs.
    """
    unzip_data(zip_path, data_folder)
    gt_df, obs_data, test_locs = load_data(data_folder)
    return prepare_tensor_datasets(obs_data, test_locs)

def add_grid_indices_and_date(test_locs):
    """
    Adds grid indices and date information to the test locations DataFrame.

    Args:
        test_locs (pd.DataFrame): DataFrame containing test location information.

    Returns:
        pd.DataFrame: Updated DataFrame with grid indices and date information.
    """
    date = np.array(np.datetime64('2020-03-05'))
    t = date.astype("float")

    delta_x = np.diff(np.sort(test_locs['x'].unique())).min()
    delta_y = np.diff(np.sort(test_locs['y'].unique())).min()

    x_start = test_locs['x'].min()
    x_end = test_locs['x'].max()
    x_coords = np.arange(x_start, x_end + delta_x, delta_x)

    y_start = test_locs['y'].min()
    y_end = test_locs['y'].max()
    y_coords = np.arange(y_start, y_end + delta_y, delta_y)

    test_locs['x_idxs'] = match(test_locs['x'].values, x_coords)
    test_locs['y_idxs'] = match(test_locs['y'].values, y_coords)

    test_locs['date'] = date
    test_locs['t'] = t

    return test_locs

def match(x, y, exact=True, tol=1e-9):
    """
    Matches elements of array x with elements in array y based on exact or approximate matching.

    Args:
        x (np.ndarray): Array of values to be matched.
        y (np.ndarray): Array of reference values to match against.
        exact (bool, optional): Whether to match exactly. Defaults to True.
        tol (float, optional): Tolerance for approximate matching. Defaults to 1e-9.

    Returns:
        np.ndarray: Array of indices in y that correspond to matches for x.

    Raises:
        AssertionError: If any value in x cannot be matched within the tolerance.
    """
    if exact:
        mask = x[:, None] == y
    else:
        dif = np.abs(x[:, None] - y)
        mask = dif < tol

    row_mask = mask.any(axis=1)
    assert row_mask.all(), f"{(~row_mask).sum()} not found, uniquely: {np.unique(np.array(x)[~row_mask])}"
    return np.argmax(mask, axis=1)

def create_datasets_and_loaders(spatial_X_train_torch, temporal_X_train_torch, y_train_torch, batch_size=1024):
    """
    Creates training and validation datasets and their corresponding DataLoaders.

    Args:
        spatial_X_train_torch (torch.Tensor): Normalized spatial training inputs.
        temporal_X_train_torch (torch.Tensor): Normalized temporal training inputs.
        y_train_torch (torch.Tensor): Normalized training target values.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 1024.

    Returns:
        tuple: A tuple containing the training and validation DataLoaders.
    """
    train_dataset = TensorDataset(spatial_X_train_torch, temporal_X_train_torch, y_train_torch)
    val_size = int(len(train_dataset) * 0.2)
    train_size = len(train_dataset) - val_size
    train_data, val_data = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader