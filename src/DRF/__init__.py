
def get_layer(name, input_dim, hidden_dim, output_dim, lengthscale, **kwargs):
    if name == "SquaredExponential":
        from DRF.layers import SquaredExponentialRFFLayer
        return SquaredExponentialRFFLayer(input_dim, hidden_dim, output_dim, lengthscale, **kwargs)
    elif name == "Matern":
        from DRF.layers import MaternRFFLayer
        return MaternRFFLayer(input_dim, hidden_dim, output_dim, lengthscale, **kwargs)
    else:
        raise NotImplementedError(f"Layer with name '{name}' is not implemented.")
