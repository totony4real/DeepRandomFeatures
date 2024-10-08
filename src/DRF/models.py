import torch
import torch.nn as nn
from DRF import get_layer
import torch.nn as nn


class DeepSpatiotemporalGPNN(nn.Module):
    def __init__(self, num_layers, spatial_input_dim, temporal_input_dim, hidden_dim, bottleneck_dim, output_dim,
                spatial_rff_layer_type, temporal_rff_layer_type,
                spatial_lengthscale, temporal_lengthscale, layer_kwargs, combine_type='concat'):
        super(DeepSpatiotemporalGPNN, self).__init__()

        self.spatial_layers = nn.ModuleList()
        self.temporal_layers = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                layer_args = (spatial_input_dim, hidden_dim, bottleneck_dim)
            elif i == num_layers - 1:
                layer_args = (bottleneck_dim+spatial_input_dim, hidden_dim, bottleneck_dim)
            else:
                layer_args = (bottleneck_dim+spatial_input_dim, hidden_dim, bottleneck_dim)

            spatial_layer = get_layer(spatial_rff_layer_type, *layer_args, lengthscale=spatial_lengthscale, **layer_kwargs)
            self.spatial_layers.append(spatial_layer)


        # Initialize only one temporal layer
        layer_args = (temporal_input_dim, hidden_dim, bottleneck_dim)
        temporal_layer = get_layer(temporal_rff_layer_type, *layer_args, lengthscale=temporal_lengthscale, **layer_kwargs)
        self.temporal_layers.append(temporal_layer)
        self.combine_type = combine_type
        self.output_layer = nn.Linear(2*bottleneck_dim, output_dim) if combine_type=='concat' else nn.Linear(bottleneck_dim, output_dim)

    def forward(self, spatial_input, temporal_input):
        spatial_x = torch.clone(spatial_input)
        for n,layer in enumerate(self.spatial_layers):
            spatial_x = layer(spatial_x)
            if n < len(self.spatial_layers)-1:
                spatial_x = torch.cat((spatial_x, spatial_input), dim=1)


        temporal_x = torch.clone(temporal_input)
        temporal_x= self.temporal_layers[0](temporal_x)

        if spatial_x.shape != temporal_x.shape:
            raise ValueError("Spatial and Temporal output dimensions do not match.")

        if self.combine_type == 'concat':
            combined_input = torch.cat((spatial_x, temporal_x), dim=1)
        elif self.combine_type == 'product':
            combined_input = spatial_x * temporal_x
        elif self.combine_type == 'sum':
            combined_input = spatial_x + temporal_x
        else:
            raise NotImplementedError

        # combined_input *= torch.sqrt(1 / combined_input.size(1))
        # combined_input *= torch.sqrt(torch.tensor(1.0 / combined_input.size(1), dtype=combined_input.dtype, device=combined_input.device))

        combined_output = self.output_layer(combined_input)
        
        return combined_output