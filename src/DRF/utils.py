import torch
import torch.nn as nn
import gc


def calculate_batch_nll(
    mu,
    var,
    y_true,
    spatial_X,
    temporal_X,
    model,
    batch_size=900,
    delta=0.1,
    gradient_penalty_weight=0.5,
):
    """
    Calculates the negative log-likelihood (NLL) for a batch of predictions using a Gaussian distribution,
    with optional gradient regularization.

    Args:
        mu (torch.Tensor): The predicted means for each data point.
        var (torch.Tensor): The predicted variances for each data point.
        y_true (torch.Tensor): The ground truth target values.
        spatial_X (torch.Tensor): The spatial input data for the model.
        temporal_X (torch.Tensor): The temporal input data for the model.
        model (torch.nn.Module): The model to compute the regularization for.
        batch_size (int, optional): The batch size for processing the data in chunks. Defaults to 900.
        delta (float, optional): The delta parameter for the Huber loss. Defaults to 0.1.
        gradient_penalty_weight (float, optional): The weight for the gradient regularization term. Defaults to 0.5.

    Returns:
        float: The total loss, combining NLL and optional gradient regularization.
        float: The average negative log-likelihood (NLL) for the provided data.
    """
    device = mu.device
    gaussian_nll_huber_loss = nn.GaussianNLLLoss(
        full=False, eps=1e-6, reduction="mean"
    ).to(device)
    total_nll = 0.0
    total_batches = 0

    for i in range(0, len(mu), batch_size):
        end_index = i + batch_size if (i + batch_size) < len(mu) else len(mu)
        mu_batch = mu[i:end_index].to(device)
        var_batch = var[i:end_index].to(device)
        y_true_batch = y_true[i:end_index].to(device)

        nll = gaussian_nll_huber_loss(mu_batch, y_true_batch, var_batch)
        total_nll += nll.item() * (end_index - i)
        total_batches += end_index - i

    nll_loss = total_nll / total_batches

    if gradient_penalty_weight > 0:
        num_samples = spatial_X.size(0)
        gradient_regularization = 0.0

        model = model.to(device)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            spatial_batch = spatial_X[start_idx:end_idx].to(device).requires_grad_(True)
            temporal_batch = (
                temporal_X[start_idx:end_idx].to(device).requires_grad_(True)
            )

            preds = model(spatial_batch, temporal_batch)

            grad_spatial = torch.autograd.grad(
                outputs=preds,
                inputs=spatial_batch,
                grad_outputs=torch.ones_like(preds),
                create_graph=True,
            )[0]
            grad_temporal = torch.autograd.grad(
                outputs=preds,
                inputs=temporal_batch,
                grad_outputs=torch.ones_like(preds),
                create_graph=True,
            )[0]

            grad_spatial_norm2 = grad_spatial.pow(2).sum()
            grad_temporal_norm2 = grad_temporal.pow(2).sum()

            gradient_regularization += (grad_spatial_norm2 + grad_temporal_norm2).item()

            del grad_spatial, grad_temporal, preds, spatial_batch, temporal_batch
            torch.cuda.empty_cache()
            gc.collect()

        gradient_regularization = gradient_penalty_weight * (
            gradient_regularization / num_samples
        )
        print("Gradient Regularization:", gradient_regularization)

        total_loss = (1 - gradient_penalty_weight) * nll_loss + gradient_regularization
    else:
        total_loss = nll_loss

    print("NLL Loss:", nll_loss)
    return total_loss, nll_loss


def functional_regularisation_S2_batched(model, dataloader, d_phi, d_theta):
    """
    Computes the spherical functional regularisation term using minibatches for memory efficiency.
    Inputs must have requires_grad=True and coordinates must be lon-lat.

    dϕ: increment of longitude
    dθ: increment of latitude

    Args:
        model (torch.nn.Module): The model to compute the regularization for.
        dataloader (torch.utils.data.DataLoader): DataLoader providing batches of data.
        d_phi (float): Increment of longitude in radians.
        d_theta (float): Increment of latitude in radians.

    Returns:
        float: The computed functional regularization term.
    """
    device = next(model.parameters()).device
    assert (
        hasattr(model, "lon_lat_inputs") and model.lon_lat_inputs
    ), "Inputs to model must be lon-lat"

    norm = 0.0
    for batch in dataloader:
        spatial_input, temporal_input, _ = batch
        spatial_input = spatial_input.clone().detach().requires_grad_(True).to(device)
        temporal_input = temporal_input.to(device)
        y = model(spatial_input, temporal_input)
        grad_outputs = torch.ones_like(y)
        grads = torch.autograd.grad(y, spatial_input, grad_outputs, create_graph=False)[
            0
        ]
        cos_theta = torch.cos(spatial_input[:, 1])
        dS = cos_theta * d_phi * d_theta
        norm += torch.sum(
            dS * (cos_theta ** (-2) * grads[:, 0] ** 2 + grads[:, 1] ** 2)
        ).item()
    return norm
