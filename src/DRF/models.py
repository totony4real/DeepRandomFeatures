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
    
    def __init__(self, num_layers, spatial_input_dim, temporal_input_dim, hidden_dim, bottleneck_dim, output_dim, combine_type='concat'):
        super(SpatiotemporalModelBase, self).__init__()
        self.num_layers = num_layers
        self.combine_type = combine_type
        self.output_dim = output_dim
        self.spatial_layers = nn.ModuleList()
        self.temporal_layers = nn.ModuleList()
        self.output_layer = nn.Linear(2 * bottleneck_dim, output_dim) if combine_type == 'concat' else nn.Linear(bottleneck_dim, output_dim)

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
        if self.combine_type == 'concat':
            return torch.cat((spatial_x, temporal_x), dim=1)
        elif self.combine_type == 'product':
            return spatial_x * temporal_x
        elif self.combine_type == 'sum':
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
    """
    
    def __init__(self, num_layers, spatial_input_dim, temporal_input_dim, hidden_dim, bottleneck_dim, output_dim, spatial_rff_layer_type, temporal_rff_layer_type, spatial_lengthscale, temporal_lengthscale, layer_kwargs, combine_type='concat'):
        super().__init__(num_layers, spatial_input_dim, temporal_input_dim, hidden_dim, bottleneck_dim, output_dim, combine_type)
        
        for i in range(num_layers):
            layer_args = (spatial_input_dim if i == 0 else bottleneck_dim, hidden_dim, bottleneck_dim)
            spatial_layer = get_layer(spatial_rff_layer_type, *layer_args, lengthscale=spatial_lengthscale, **layer_kwargs)
            self.spatial_layers.append(spatial_layer)

        temporal_layer = get_layer(temporal_rff_layer_type, temporal_input_dim, hidden_dim, bottleneck_dim, lengthscale=temporal_lengthscale, **layer_kwargs)
        self.temporal_layers.append(temporal_layer)

    def process_spatial(self, spatial_input):
        """
        Processes spatial input through a series of spatial layers.

        Args:
            spatial_input (torch.Tensor): Spatial input tensor.

        Returns:
            torch.Tensor: Processed spatial data.
        """
        for layer in self.spatial_layers:
            spatial_input = torch.cat((spatial_input, layer(spatial_input)), dim=1)
        return spatial_input

    def process_temporal(self, temporal_input):
        """
        Processes temporal input through the temporal layer.

        Args:
            temporal_input (torch.Tensor): Temporal input tensor.

        Returns:
            torch.Tensor: Processed temporal data.
        """
        return self.temporal_layers[0](temporal_input)


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
    
    def __init__(self, num_layers, spatial_input_dim, temporal_input_dim, hidden_dim, bottleneck_dim, output_dim, spatial_lengthscale, temporal_lengthscale, nu, amplitude, lengthscale2, amplitude2, lon_lat_inputs=True, combine_type='concat'):
        super().__init__(num_layers, spatial_input_dim, temporal_input_dim, hidden_dim, bottleneck_dim, output_dim, combine_type)

        s2_dim = 2 if lon_lat_inputs else 3

        for i in range(num_layers):
            if i == 0:
                layer = MaternRandomPhaseS2RFFLayer(
                    hidden_dim=hidden_dim,
                    output_dim=bottleneck_dim,
                    lengthscale=spatial_lengthscale,
                    nu=nu,
                    amplitude=amplitude,
                    lon_lat_inputs=lon_lat_inputs)
            else:
                layer = self.SumFeatures(
                    input_dim=bottleneck_dim + s2_dim,
                    hidden_dim=hidden_dim,
                    output_dim=bottleneck_dim,
                    lengthscale=spatial_lengthscale,
                    amplitude=amplitude,
                    lengthscale2=lengthscale2,
                    amplitude2=amplitude2,
                    nu=nu,
                    lon_lat_inputs=lon_lat_inputs)
            self.spatial_layers.append(layer)

        temporal_layer = MaternRFFLayer(temporal_input_dim, hidden_dim, bottleneck_dim, temporal_lengthscale, nu, amplitude)
        self.temporal_layers.append(temporal_layer)
    
    def process_spatial(self, spatial_input):
        """
        Processes spatial input through the series of spatial layers.

        Args:
            spatial_input (torch.Tensor): Spatial input tensor.

        Returns:
            torch.Tensor: Processed spatial data.
        """
        for layer in self.spatial_layers:
            spatial_input = layer(spatial_input)
        return spatial_input

    def process_temporal(self, temporal_input):
        """
        Processes temporal input through the temporal layer.

        Args:
            temporal_input (torch.Tensor): Temporal input tensor.

        Returns:
            torch.Tensor: Processed temporal data.
        """
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
        
        def __init__(self, input_dim, hidden_dim, output_dim, lengthscale=1., nu=3/2, amplitude=1., lengthscale2=1., amplitude2=1., lon_lat_inputs=True):
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
            self.hidden_layer_rn, self.hidden_layer_s2, self.output_layer = self.initialize_layers()

        def initialize_layers(self):
            """
            Initializes layers based on Matern RFF and random phase feature maps.

            Returns:
                Tuple[nn.Linear, nn.Module, nn.Linear]: Initialized hidden and output layers.
            """
            input_dim_x = self.input_dim - 2 if self.lon_lat_inputs else self.input_dim - 3
            
            hidden_layer_rn = MaternRFFLayer(input_dim_x, self.hidden_dim, self.output_dim, self.lengthscale2, self.nu, self.amplitude2).hidden_layer
            hidden_layer_s2 = MaternRandomPhaseS2RFFLayer(self.hidden_dim, self.output_dim, self.lengthscale, self.nu, self.amplitude).feature_map
            output_layer = nn.Linear(self.hidden_dim, self.output_dim, bias=False)

            return hidden_layer_rn, hidden_layer_s2, output_layer

        @staticmethod
        def spherical_to_cartesian(lon, lat):
            """
            Converts spherical coordinates (longitude and latitude) to Cartesian coordinates.

            Args:
                lon (torch.Tensor): Longitude values in radians.
                lat (torch.Tensor): Latitude values in radians.

            Returns:
                torch.Tensor: A tensor with Cartesian coordinates.
            """
            x = torch.cos(lat) * torch.cos(lon)
            y = torch.cos(lat) * torch.sin(lon)
            z = torch.sin(lat)
            return torch.stack((x, y, z), dim=1)

        def forward(self, x):
            """
            Processes input through the hidden layers and combines the features.

            Args:
                x (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: Processed output tensor.
            """
            x_ = x[:, :-2] if self.lon_lat_inputs else x[:, :-3]
            s = x[:, -2:] if self.lon_lat_inputs else x[:, -3:]
            s = self.spherical_to_cartesian(s[:, 0], s[:, 1]) if self.lon_lat_inputs else s
            
            x = self.hidden_layer_rn(x_)
            scaling_factor = torch.sqrt(torch.tensor(2.0 * self.amplitude**2 / self.hidden_layer_rn.out_features))
            x = scaling_factor * torch.cos(x)
            x += self.hidden_layer_s2(s)
            x = self.output_layer(x)
            return x
