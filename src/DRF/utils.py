import torch

def calculate_batch_nll(mu, var, y_true, batch_size=1024):
    """
    Calculates the negative log-likelihood (NLL) for a batch of predictions using a Gaussian distribution.

    Args:
        mu (torch.Tensor): The predicted means for each data point.
        var (torch.Tensor): The predicted variances for each data point.
        y_true (torch.Tensor): The ground truth target values.
        batch_size (int, optional): The batch size for processing the data in chunks. Defaults to 1024.

    Returns:
        float: The average negative log-likelihood (NLL) for the provided data.
    """
    gaussian_nll_loss = torch.nn.GaussianNLLLoss(full=False, eps=1e-6, reduction='mean').to(mu.device)
    total_nll = 0.0
    total_batches = 0

    for i in range(0, len(mu), batch_size):
        end_index = i + batch_size if (i + batch_size) < len(mu) else len(mu)
        mu_batch = mu[i:end_index].to(mu.device)
        var_batch = var[i:end_index].to(mu.device)
        y_true_batch = y_true[i:end_index].to(mu.device)

        nll = gaussian_nll_loss(mu_batch, y_true_batch.unsqueeze(1), var_batch)
        total_nll += nll.item() * (end_index - i)
        total_batches += (end_index - i)

    return total_nll / total_batches
