#%%
import torch
import torch.nn as nn
import numpy as np
import geometric_kernels.torch
from torch import lgamma, exp
from abc import ABC, abstractmethod
from scipy.special import gegenbauer as scipy_gegenbauer
from geometric_kernels.spaces import Hypersphere
from geometric_kernels.kernels import MaternGeometricKernel, default_feature_map


class RFFLayer(nn.Module, ABC):
    """
    A base class for Random Fourier Feature (RFF) layers. This layer uses random Fourier 
    features to approximate kernel functions. Subclasses should implement the 
    `sample_from_spectral_density` method to specify the sampling from the spectral 
    density corresponding to a particular kernel.

    Args:
        input_dim (int): The dimensionality of the input data.
        hidden_dim (int): The number of random features (hidden layer size).
        output_dim (int): The dimensionality of the output layer.
        amplitude (float, optional): A scaling factor for the layer output. Defaults to 1.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, amplitude=1.):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.amplitude = amplitude
        self.hidden_layer, self.output_layer = self.initialize_layers()

    @abstractmethod
    def sample_from_spectral_density(self, shape):
        """
        Samples weights from the spectral density associated with the kernel.
        
        Args:
            shape (torch.Size): The shape of the tensor to be sampled.
        
        Returns:
            torch.Tensor: A tensor with the sampled values.
        
        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def initialize_layers(self):
        """
        Initializes the hidden and output layers. The hidden layer weights are sampled 
        from the spectral density specified by the subclass, while biases are initialized 
        uniformly. The output layer weights are initialized with a normal distribution.

        Returns:
            Tuple[nn.Linear, nn.Linear]: The initialized hidden and output linear layers.
        """
        hidden_layer = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        output_layer = nn.Linear(self.hidden_dim, self.output_dim, bias=True)

        hidden_layer.weight.data = self.sample_from_spectral_density(hidden_layer.weight.shape)
        hidden_layer.bias.data.uniform_(0, 2 * np.pi)
        output_layer.weight.data.normal_(0, 1)

        hidden_layer.weight.requires_grad = False
        hidden_layer.bias.requires_grad = False

        return hidden_layer, output_layer

    def forward(self, x):
        """
        The forward pass through the RFF layer, applying random Fourier features 
        and a cosine activation before passing through the output layer.

        Args:
            x (torch.Tensor): Input tensor of shape (N, input_dim).

        Returns:
            torch.Tensor: The transformed output tensor of shape (N, output_dim).
        """
        x = self.hidden_layer(x)
        scaling_factor = torch.sqrt(torch.tensor(2.0 * self.amplitude**2 / self.hidden_layer.out_features))
        x = scaling_factor * torch.cos(x)
        x = self.output_layer(x)
        return x


class SquaredExponentialRFFLayer(RFFLayer):
    """
    An implementation of RFFLayer for the Squared Exponential (Gaussian) kernel. 
    Uses a normal distribution to sample from the spectral density.

    Args:
        input_dim (int): The dimensionality of the input data.
        hidden_dim (int): The number of random features.
        output_dim (int): The dimensionality of the output layer.
        lengthscale (float, optional): The length scale parameter of the kernel. Defaults to 1.
        amplitude (float, optional): A scaling factor for the layer output. Defaults to 1.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, lengthscale=1., amplitude=1.):
        self.lengthscale = lengthscale
        super().__init__(input_dim, hidden_dim, output_dim, amplitude)

    def sample_from_spectral_density(self, shape):
        """
        Samples weights from the spectral density of the Squared Exponential kernel, 
        which is a normal distribution.

        Args:
            shape (torch.Size): The shape of the tensor to be sampled.
        
        Returns:
            torch.Tensor: A tensor with the sampled values from a normal distribution.
        """
        return torch.distributions.Normal(loc=0, scale=1/self.lengthscale).sample(shape)


class MaternRFFLayer(RFFLayer):
    """
    An implementation of RFFLayer for the Matern kernel. Uses a Student's t-distribution 
    to sample from the spectral density, which is appropriate for the Matern kernel.

    Args:
        input_dim (int): The dimensionality of the input data.
        hidden_dim (int): The number of random features.
        output_dim (int): The dimensionality of the output layer.
        lengthscale (float, optional): The length scale parameter of the kernel. Defaults to 1.
        nu (float, optional): The smoothness parameter of the Matern kernel. Defaults to 3/2.
        amplitude (float, optional): A scaling factor for the layer output. Defaults to 1.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, lengthscale=1., nu=3/2, amplitude=1.):
        self.lengthscale = lengthscale
        self.nu = nu
        super().__init__(input_dim, hidden_dim, output_dim, amplitude)

    def sample_from_spectral_density(self, shape):
        """
        Samples weights from the spectral density of the Matern kernel, 
        using a Student's t-distribution.

        Args:
            shape (torch.Size): The shape of the tensor to be sampled.
        
        Returns:
            torch.Tensor: A tensor with the sampled values from a Student's t-distribution.
        """
        return torch.distributions.StudentT(df=2*self.nu, scale=1/self.lengthscale).sample(shape)


class RandomPhaseFeatureMap:
    """
    Random phase feature maps for spherical Matern GP 
    """

    def __init__(self, num_features, nu, lengthscale, num_levels=24):
        sphere = Hypersphere(dim=2)
        kernel = MaternGeometricKernel(sphere, num=num_levels)
        self.N = num_features
        spectrum = kernel._spectrum(sphere.get_eigenvalues(num_levels),
                                    nu=torch.tensor([nu]),
                                    lengthscale=torch.tensor([lengthscale]))
        self.spectrum = spectrum / torch.sum(spectrum)

        self.gegenbauer_coeff_table = torch.zeros(num_levels, num_levels + 1)
        for i in range(num_levels):
            coeffs = scipy_gegenbauer(i, 1/2).coefficients
            self.gegenbauer_coeff_table[i][-(i+1):] = torch.tensor(coeffs)

        self.levels = torch.multinomial(self.spectrum.squeeze(), num_features, replacement=True)
        noise = torch.randn(num_features, 3)
        self.noise = noise / torch.linalg.norm(noise, dim=1, keepdim=True)


    @staticmethod
    def polyval(coeffs, x):
        """
        Compute batched values of a polynomial.
        """
        assert len(coeffs) == len(x)
        curVal = torch.zeros_like(x)
        for i in range(len(coeffs.T) - 1):
            curVal = (curVal + coeffs[:, [i]]) * x
        return curVal + coeffs[:, [-1]]

    def get_gegenbauer12_coeffs(self, n):
        """Get coefficients of Gegenbauer polynomial with α=1/2 and order n"""
        return self.gegenbauer_coeff_table[n]

    def gegenbauer12(self, n, x):
        """Vectorised evaluation of Gegenbauer polynomial with α=1/2 and orders n₁,...,nₕ"""
        coeffs = self.get_gegenbauer12_coeffs(n)
        return self.polyval(coeffs, x)

    @staticmethod
    def gamma(x):
        return torch.exp(torch.lgamma(x))

    def c(self, l):
        c = (2 * l + 1) / (4 * torch.pi)
        return c[:, None]

    def __call__(self, X):
        device = X.device
        self.noise = self.noise.to(device)
        self.levels = self.levels.to(device)
        self.gegenbauer_coeff_table = self.gegenbauer_coeff_table.to(device)
        
        UXT = self.noise @ X.T  # Shape (N, B)
        c = self.c(self.levels)
        fX = torch.sqrt(c) * self.gegenbauer12(self.levels, UXT) / np.sqrt(self.N * 0.079)  # Shape (N, B)
        return fX.T  # Shape (B, N)


def spherical_to_cartesian(lon, lat):
    """
        Converts spherical coordinates (longitude and latitude) to Cartesian coordinates.

        Args:
            lon (torch.Tensor): Longitude values in radians.
            lat (torch.Tensor): Latitude values in radians.

        Returns:
            torch.Tensor: A tensor containing the Cartesian coordinates, with shape (N, 3).
    """
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.stack((x, y, z), dim=1)

class MaternRandomPhaseS2RFFLayer(nn.Module):
    """
    A neural network layer that applies a random phase feature map based on 
    spherical Matern Gaussian Processes, particularly useful for data in spherical 
    coordinates such as longitude and latitude.

    Args:
        hidden_dim (int): The dimensionality of the hidden layer.
        output_dim (int): The dimensionality of the output layer.
        lengthscale (float, optional): The length scale parameter for the Matern kernel. 
            Defaults to 1.
        nu (float, optional): The smoothness parameter of the Matern kernel. Common values 
            are 1/2, 3/2, or 5/2. Defaults to 3/2.
        amplitude (float, optional): A scaling factor applied to the output of the feature map. 
            Defaults to 1.
        lon_lat_inputs (bool, optional): Indicates whether the input is in longitude and latitude 
            format. If True, inputs are converted to Cartesian coordinates. Defaults to True.
    """
    def __init__(self, hidden_dim, output_dim, lengthscale=1., nu=3/2, amplitude=1., lon_lat_inputs=True):
        super().__init__()
        self.lengthscale = lengthscale
        self.nu = nu
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.amplitude = amplitude
        self.lon_lat_inputs = lon_lat_inputs
        self.feature_map = self.initialize_feature_map()
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        self.output_layer.weight.data.normal_(0, 1)

    def initialize_feature_map(self):
        
        feature_map = RandomPhaseFeatureMap(num_features=self.hidden_dim,
                                            nu=self.nu,
                                            lengthscale=self.lengthscale)
        return feature_map

    def forward(self, x):
        """
        The forward pass of the layer, applying the feature map and linear transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 2) for longitude-latitude coordinates 
                or shape (N, 3) if already in Cartesian coordinates.

        Returns:
            torch.Tensor: The transformed output tensor of shape (N, output_dim).
        """
        # Ensure input `x` is on the same device as the model parameters
        device = next(self.parameters()).device
        x = x.to(device)
        if self.lon_lat_inputs:
            x = spherical_to_cartesian(x[:,0], x[:,1])
        x = self.feature_map(x)
        x *= self.amplitude
        x = self.output_layer(x)
        return x





# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm, colors

    # Test random phase S2 Matern network
    # define latitude and longitude discretization
    _NUM_LATS = 128
    _NUM_LONGS = 128

    # generate a grid on the sphere
    x = torch.linspace(0, 2*torch.pi, _NUM_LATS)
    y = torch.linspace(0, torch.pi, _NUM_LONGS)
    lats, longs =  torch.meshgrid(x, y)

    other_points_xs = torch.sin(longs) * torch.cos(lats)
    other_points_ys = torch.sin(longs) * torch.sin(lats)
    other_points_zs = torch.cos(longs)

    other_points = torch.stack((torch.ravel(other_points_xs),
                                torch.ravel(other_points_ys),
                                torch.ravel(other_points_zs)), dim=1)

    # Initialise random phase S2 Matern network
    model = MaternRandomPhaseS2RFFLayer(hidden_dim=1000,
                                        output_dim=1,
                                        lengthscale=0.3,
                                        nu=3/2)

    # Plot output at initialisation
    initial_sample = model(other_points).detach().numpy() 

    fig = plt.figure()
    cmap = plt.get_cmap('viridis')
    ax = fig.add_subplot(111, projection='3d')
    norm=colors.Normalize(vmin = np.min(initial_sample),
                        vmax = np.max(initial_sample), clip = False)
    surf = ax.plot_surface(other_points_xs, other_points_ys, other_points_zs,
                            facecolors=cmap(norm(initial_sample.reshape(_NUM_LATS, _NUM_LONGS))), 
                            cstride=1, rstride=1) 


# %%
