import torch
import torch.nn as nn
from DRF import get_layer
from DRF.layers import MaternRandomPhaseS2RFFLayer, MaternRFFLayer


class SpatiotemporalModelBase(nn.Module):
    """
    A base class for spatiotemporal models, which defines a structure for models that
    process spatial and temporal data independently and then combine them in various ways.

    Args:
        num_layers (int): The number of spatial layers.
        spatial_input_dim (int): Dimensionality of spatial input data.
        temporal_input_dim (int): Dimensionality of temporal input data.
        hidden_dim (int): Dimensionality of hidden layers.
        bottleneck_dim (int): Dimensionality of bottleneck layers.
        output_dim (int): Dimensionality of the model's output.
        combine_type (str, optional): Method for combining spatial and temporal features.
            Options are 'concat', 'product', or 'sum'. Defaults to 'concat'.
    """

    def __init__(
        self,
        num_layers,
        spatial_input_dim,
        temporal_input_dim,
        hidden_dim,
        bottleneck_dim,
        output_dim,
        combine_type="concat",
    ):
        super(SpatiotemporalModelBase, self).__init__()
        self.num_layers = num_layers
        self.combine_type = combine_type
        self.output_dim = output_dim
        self.spatial_layers = nn.ModuleList()
        self.temporal_layers = nn.ModuleList()
        self.output_layer = (
            nn.Linear(2 * bottleneck_dim, output_dim)
            if combine_type == "concat"
            else nn.Linear(bottleneck_dim, output_dim)
        )

    def forward(self, spatial_input, temporal_input):
        """
        The forward pass that processes spatial and temporal inputs and combines them.

        Args:
            spatial_input (torch.Tensor): Spatial input tensor.
            temporal_input (torch.Tensor): Temporal input tensor.

        Returns:
            torch.Tensor: The final output after combining spatial and temporal data.
        """
        spatial_x = self.process_spatial(spatial_input)
        temporal_x = self.process_temporal(temporal_input)

        if spatial_x.shape != temporal_x.shape:
            raise ValueError("Spatial and Temporal output dimensions do not match.")

        combined_output = self.combine_features(spatial_x, temporal_x)
        return self.output_layer(combined_output)

    def process_spatial(self, spatial_input):
        """
        Placeholder method to process spatial input. This should be implemented in subclasses.

        Args:
            spatial_input (torch.Tensor): Spatial input tensor.

        Returns:
            torch.Tensor: Processed spatial data.
        """
        return spatial_input

    def process_temporal(self, temporal_input):
        """
        Placeholder method to process temporal input. This should be implemented in subclasses.

        Args:
            temporal_input (torch.Tensor): Temporal input tensor.

        Returns:
            torch.Tensor: Processed temporal data.
        """
        return temporal_input

    def combine_features(self, spatial_x, temporal_x):
        """
        Combines spatial and temporal features according to the specified `combine_type`.

        Args:
            spatial_x (torch.Tensor): Processed spatial data.
            temporal_x (torch.Tensor): Processed temporal data.

        Returns:
            torch.Tensor: Combined spatial and temporal data.
        """
        if self.combine_type == "concat":
            return torch.cat((spatial_x, temporal_x), dim=1)
        elif self.combine_type == "product":
            return spatial_x * temporal_x
        elif self.combine_type == "sum":
            return spatial_x + temporal_x
        else:
            raise NotImplementedError


class DeepSpatiotemporalGPNN(SpatiotemporalModelBase):
    """
    A spatiotemporal neural network that utilizes random Fourier features for processing
    both spatial and temporal data separately, then combines them.

    Args:
        spatial_rff_layer_type (str): Type of RFF layer to use for spatial data.
        temporal_rff_layer_type (str): Type of RFF layer to use for temporal data.
        spatial_lengthscale (float): Lengthscale parameter for spatial layers.
        temporal_lengthscale (float): Lengthscale parameter for temporal layers.
        layer_kwargs (dict): Additional arguments for layer initialization.
        combine_type (str, optional): Method for combining spatial and temporal features.
        device (str, optional): Device to run the model on. Defaults to 'cuda'.
    """

    def __init__(
        self,
        num_layers,
        spatial_input_dim,
        temporal_input_dim,
        hidden_dim,
        bottleneck_dim,
        output_dim,
        spatial_rff_layer_type,
        temporal_rff_layer_type,
        spatial_lengthscale,
        temporal_lengthscale,
        layer_kwargs,
        combine_type="concat",
        device="cuda",
    ):
        super().__init__(
            num_layers,
            spatial_input_dim,
            temporal_input_dim,
            hidden_dim,
            bottleneck_dim,
            output_dim,
            combine_type,
        )
        self.device = device

        for i in range(num_layers):
            if i == 0:
                layer_args = (spatial_input_dim, hidden_dim, bottleneck_dim)
            else:
                layer_args = (
                    bottleneck_dim + spatial_input_dim,
                    hidden_dim,
                    bottleneck_dim,
                )
            spatial_layer = get_layer(
                spatial_rff_layer_type,
                *layer_args,
                lengthscale=spatial_lengthscale,
                **layer_kwargs,
            )
            self.spatial_layers.append(spatial_layer)

        layer_args = (temporal_input_dim, hidden_dim, bottleneck_dim)
        temporal_layer = get_layer(
            temporal_rff_layer_type,
            *layer_args,
            lengthscale=temporal_lengthscale,
            **layer_kwargs,
        )
        self.temporal_layers.append(temporal_layer)

    def process_spatial(self, spatial_input):
        """
        Processes spatial input through a series of spatial layers.

        Args:
            spatial_input (torch.Tensor): Spatial input tensor.

        Returns:
            torch.Tensor: Processed spatial data.
        """
        spatial_x = spatial_input.clone()
        for n, layer in enumerate(self.spatial_layers):
            spatial_x = layer(spatial_x)
            if n < len(self.spatial_layers) - 1:
                spatial_x = torch.cat((spatial_x, spatial_input), dim=1)
        return spatial_x

    def process_temporal(self, temporal_input):
        """
        Processes temporal input through the temporal layer.

        Args:
            temporal_input (torch.Tensor): Temporal input tensor.

        Returns:
            torch.Tensor: Processed temporal data.
        """
        temporal_x = self.temporal_layers[0](temporal_input)
        return temporal_x


def initialize_model(
    model_name,
    num_layers,
    spatial_input_dim,
    temporal_input_dim,
    hidden_dim,
    bottleneck_dim,
    output_dim,
    spatial_lengthscale,
    temporal_lengthscale,
    amplitude,
    device,
    spatial_layer_type="Matern",
    temporal_layer_type="Matern",
    model_kwargs=None,
):
    """
    Initializes the specified model based on the provided configuration.

    Args:
        model_name (str): The name of the model to initialize. Supported options are 'DeepSpatiotemporalGPNN' and 'DeepMaternRandomPhaseS2RFFNN'.
        num_layers (int): The number of layers in the model.
        spatial_input_dim (int): The dimensionality of the spatial input.
        temporal_input_dim (int): The dimensionality of the temporal input.
        hidden_dim (int): The dimensionality of the hidden layers.
        bottleneck_dim (int): The dimensionality of the bottleneck layer.
        output_dim (int): The dimensionality of the output.
        spatial_lengthscale (float): The lengthscale for the spatial input.
        temporal_lengthscale (float): The lengthscale for the temporal input.
        amplitude (float): The amplitude used in the model.
        device (torch.device): The device (CPU or GPU) on which the model will be run.
        spatial_layer_type (str, optional): The type of spatial layer to use. Default is 'Matern'.
        temporal_layer_type (str, optional): The type of temporal layer to use. Default is 'Matern'.
        model_kwargs (dict, optional): Additional keyword arguments to pass to the model. Default is None.

    Returns:
        torch.nn.Module: The initialized model.

    Raises:
        ValueError: If an unsupported model name is provided.
    """

    if model_kwargs is None:
        model_kwargs = {}
    if model_name == "DeepSpatiotemporalGPNN":
        layer_kwargs = {"amplitude": amplitude}
        return DeepSpatiotemporalGPNN(
            num_layers=num_layers,
            spatial_input_dim=spatial_input_dim,
            temporal_input_dim=temporal_input_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            output_dim=output_dim,
            spatial_rff_layer_type=spatial_layer_type,
            temporal_rff_layer_type=temporal_layer_type,
            spatial_lengthscale=spatial_lengthscale,
            temporal_lengthscale=temporal_lengthscale,
            layer_kwargs=layer_kwargs,
            device=device,
            **model_kwargs,
        )
    elif model_name == "DeepMaternRandomPhaseS2RFFNN":
        return DeepMaternRandomPhaseS2RFFNN(
            num_layers=num_layers,
            spatial_input_dim=spatial_input_dim,
            temporal_input_dim=temporal_input_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            output_dim=output_dim,
            spatial_lengthscale=spatial_lengthscale,
            temporal_lengthscale=temporal_lengthscale,
            amplitude=amplitude,
            device=device,
            **model_kwargs,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


class DeepMaternRandomPhaseS2RFFNN(SpatiotemporalModelBase):
    """
    A spatiotemporal neural network utilizing Matern random phase features for spherical
    data, specifically designed for processing spatial data on a spherical domain.

    Args:
        spatial_lengthscale (float): Lengthscale parameter for spatial layers.
        temporal_lengthscale (float): Lengthscale parameter for temporal layers.
        nu (float): Smoothness parameter for the Matern kernel.
        amplitude (float): Amplitude scaling for spatial layers.
        lengthscale2 (float): Lengthscale parameter for additional spatial layers.
        amplitude2 (float): Additional amplitude scaling.
        lon_lat_inputs (bool, optional): If True, input is expected to be longitude and latitude.
        combine_type (str, optional): Method for combining spatial and temporal features.
    """

    def __init__(
        self,
        num_layers,
        spatial_input_dim,
        temporal_input_dim,
        hidden_dim,
        bottleneck_dim,
        output_dim,
        spatial_lengthscale,
        temporal_lengthscale,
        nu,
        amplitude,
        lengthscale2,
        amplitude2,
        lon_lat_inputs=True,
        combine_type="concat",
        device="cpu",
    ):
        super().__init__(
            num_layers,
            spatial_input_dim,
            temporal_input_dim,
            hidden_dim,
            bottleneck_dim,
            output_dim,
            combine_type,
        )
        self.device = device
        self.lon_lat_inputs = lon_lat_inputs
        self.to(self.device)

        input_dim = spatial_input_dim
        s2_dim = 2 if lon_lat_inputs else 3
        for i in range(num_layers):
            if i == 0:
                layer = MaternRandomPhaseS2RFFLayer(
                    hidden_dim=hidden_dim,
                    output_dim=bottleneck_dim,
                    lengthscale=spatial_lengthscale,
                    nu=nu,
                    amplitude=amplitude,
                    lon_lat_inputs=lon_lat_inputs,
                )
            else:
                layer = SumFeatures(
                    input_dim=bottleneck_dim + s2_dim,
                    hidden_dim=hidden_dim,
                    output_dim=bottleneck_dim,
                    lengthscale=spatial_lengthscale,
                    amplitude=amplitude,
                    lengthscale2=lengthscale2,
                    amplitude2=amplitude2,
                    nu=nu,
                    lon_lat_inputs=lon_lat_inputs,
                )
            self.spatial_layers.append(layer.to(self.device))

        self.temporal_layers = nn.ModuleList()
        temporal_layer = MaternRFFLayer(
            input_dim=temporal_input_dim,
            hidden_dim=hidden_dim,
            output_dim=bottleneck_dim,
            lengthscale=temporal_lengthscale,
            nu=nu,
            amplitude=amplitude,
        )
        self.temporal_layers.append(temporal_layer.to(self.device))

    def process_spatial(self, spatial_input):
        original_spatial_input = spatial_input
        x_spatial = original_spatial_input.clone()

        for n, layer in enumerate(self.spatial_layers):
            if n < self.num_layers - 1:
                x_spatial = layer(x_spatial)
                x_spatial = torch.cat([x_spatial, original_spatial_input], dim=1)
            else:
                x_spatial = layer(x_spatial)

        return x_spatial

    def process_temporal(self, temporal_input):
        """
        Processes temporal input through the temporal layer.

        Args:
            temporal_input (torch.Tensor): Temporal input tensor.

        Returns:
            torch.Tensor: Processed temporal data.
        """
        temporal_input = temporal_input.to(self.device)
        return self.temporal_layers[0](temporal_input)


class SumFeatures(nn.Module):
    """
    A specialized layer for combining features, tailored for spherical data with
    Matern random phase features.

    Args:
        input_dim (int): Dimensionality of the input.
        hidden_dim (int): Dimensionality of the hidden layer.
        output_dim (int): Dimensionality of the output layer.
        lengthscale (float): Lengthscale for the hidden layer.
        nu (float): Smoothness parameter for the Matern kernel.
        amplitude (float): Amplitude scaling for hidden layers.
        lengthscale2 (float): Secondary lengthscale for an additional layer.
        amplitude2 (float): Secondary amplitude scaling.
        lon_lat_inputs (bool): If True, expects longitude and latitude inputs.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        lengthscale=1.0,
        nu=3 / 2,
        amplitude=1.0,
        lengthscale2=1.0,
        amplitude2=1.0,
        lon_lat_inputs=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lengthscale = lengthscale
        self.nu = nu
        self.amplitude = amplitude
        self.lengthscale2 = lengthscale2
        self.amplitude2 = amplitude2
        self.lon_lat_inputs = lon_lat_inputs
        self.hidden_layer_rn, self.hidden_layer_s2, self.output_layer = (
            self.initialize_layers()
        )

    def initialize_layers(self):
        """
        Initializes layers based on Matern RFF and random phase feature maps.

        Returns:
            Tuple[nn.Linear, nn.Module, nn.Linear]: Initialized hidden and output layers.
        """
        if self.lon_lat_inputs:
            input_dim_x = self.input_dim - 2
        else:
            input_dim_x = self.input_dim - 3

        hidden_layer_rn = MaternRFFLayer(
            input_dim_x,
            self.hidden_dim,
            self.output_dim,
            self.lengthscale2,
            self.nu,
            self.amplitude2,
        ).hidden_layer
        hidden_layer_s2 = MaternRandomPhaseS2RFFLayer(
            self.hidden_dim, self.output_dim, self.lengthscale, self.nu, self.amplitude
        ).feature_map
        output_layer = nn.Linear(self.hidden_dim, self.output_dim, bias=False)

        return hidden_layer_rn, hidden_layer_s2, output_layer

    def forward(self, x):
        """
        Processes input through the hidden layers and combines the features.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Processed output tensor.
        """
        if self.lon_lat_inputs:
            x_ = x[:, :-2]
            s = x[:, -2:]
            s = self.spherical_to_cartesian(s[:, 0], s[:, 1])
        else:
            x_ = x[:, :-3]
            s = x[:, -3:]
        x = self.hidden_layer_rn(x_)
        scaling_factor = torch.sqrt(
            torch.tensor(2.0 * self.amplitude**2 / self.hidden_layer_rn.out_features)
        )
        x = scaling_factor * torch.cos(x)
        x += self.hidden_layer_s2(s)
        x = self.output_layer(x)
        return x

    def spherical_to_cartesian(self, lon, lat):
        """
        Processes input through the hidden layers and combines the features.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Processed output tensor.
        """
        x = torch.cos(lat) * torch.cos(lon)
        y = torch.cos(lat) * torch.sin(lon)
        z = torch.sin(lat)
        return torch.stack((x, y, z), dim=1)
