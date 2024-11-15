#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors

def spherical_to_cartesian(lon, lat):
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack((x, y, z), axis=0)

#%%
file_path = '/Users/sotakao/Dropbox/Mac/Documents/Gaussian-Process-Approximation/Data/combined_data_clean_new.csv'
df = pd.read_csv(file_path)
assert all(col in df.columns for col in ['ssha_20_ku', 'lat_20_ku', 'lon_20_ku']), "CSV does not have the required columns."

df['lon_20_ku'] = np.deg2rad(df['lon_20_ku'])
df['lat_20_ku'] = np.deg2rad(df['lat_20_ku']) #+ np.pi/2

observed_points = np.stack([df['lon_20_ku'].values, df['lat_20_ku'].values], axis=1)
observed_values = df['ssha_20_ku'].values.reshape(-1, 1)
observed_points = observed_points[::100]
observed_values = observed_values[::100]

longitudes = np.rad2deg(observed_points[:,0])
latitudes = np.rad2deg(observed_points[:,1])
c = observed_values.copy()
c[c > 0.3] = 0.3
c[c < -0.3] = -0.3

# %%
file_path = '/Users/sotakao/Dropbox/Mac/Documents/Gaussian-Process-Approximation/Data/official_results_EuclideanDRF.npy'
drf_results = np.load(file_path)

_NUM_LONGS = 512
_NUM_LATS = 256
x = np.linspace(0, 2 * np.pi, _NUM_LONGS) 
y = np.linspace(-np.pi / 2, np.pi / 2, _NUM_LATS)
longs, lats = np.meshgrid(x, y)
X = np.rad2deg(longs)
Y = np.rad2deg(lats)

# Clip values for plotting
drf_results[drf_results > 0.3] = 0.3
drf_results[drf_results < -0.3] = -0.3
Z = drf_results.reshape(_NUM_LONGS, _NUM_LATS)

fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(projection=ccrs.Orthographic(-260, -20))
crs = ccrs.RotatedPole(pole_longitude=180)
CS = plt.contourf(X, Y, Z.T, levels=50, transform=crs, cmap='coolwarm'
, vmin=-0.3, vmax=0.3
)
ax.gridlines()
ax.add_feature(cfeature.BORDERS, zorder=11, linestyle=':')
ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black')
ax.coastlines()
ax.set_global()
ax.set_title("Euclidean DRF", fontsize=20)
# cbar = fig.colorbar(CS, ax=ax, shrink=0.8, aspect=10)
# cbar.set_ticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
# cbar.set_ticklabels([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
# cbar.ax.tick_params(labelsize=20)
plt.tight_layout()
# plt.savefig('figures/sla_eulcidean_drf_predictions.pdf', bbox_inches='tight')

# %%
file_path = '/Users/sotakao/Dropbox/Mac/Documents/Gaussian-Process-Approximation/Data/official_results_sphereDRF.npy'
drf_results = np.load(file_path)

_NUM_LONGS = 512
_NUM_LATS = 256
x = np.linspace(0, 2 * np.pi, _NUM_LONGS) 
y = np.linspace(-np.pi / 2, np.pi / 2, _NUM_LATS)
longs, lats = np.meshgrid(x, y)
X = np.rad2deg(longs)
Y = np.rad2deg(lats)

# Clip values for plotting
drf_results[drf_results > 0.3] = 0.3
drf_results[drf_results < -0.3] = -0.3
Z = drf_results.reshape(_NUM_LONGS, _NUM_LATS)

fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(projection=ccrs.Orthographic(-260, -20))
crs = ccrs.RotatedPole(pole_longitude=180)
CS = plt.contourf(X, Y, Z.T, levels=50, transform=crs, cmap='coolwarm', vmin=-0.3, vmax=0.3)
ax.gridlines()
ax.add_feature(cfeature.BORDERS, zorder=11, linestyle=':')
ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black')
ax.coastlines()
ax.set_global()
ax.set_title("Spherical DRF", fontsize=20)
cbar = fig.colorbar(CS, ax=ax, shrink=0.8, aspect=10)
cbar.set_ticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
cbar.set_ticklabels([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
cbar.ax.tick_params(labelsize=20)
plt.tight_layout()
plt.savefig('figures/sla_spherical_drf_predictions.pdf', bbox_inches='tight')


# %%
file_path = '/Users/sotakao/Dropbox/Mac/Documents/Gaussian-Process-Approximation/Data/official_results_sphereSVGP.npy'
svgp_results = np.load(file_path)

_NUM_LONGS = 512
_NUM_LATS = 256
x = np.linspace(0, 2 * np.pi, _NUM_LONGS) 
y = np.linspace(-np.pi / 2, np.pi / 2, _NUM_LATS)
longs, lats = np.meshgrid(x, y)
X = np.rad2deg(longs)
Y = np.rad2deg(lats)

# Clip values for plotting
svgp_results[svgp_results > 0.3] = 0.3
svgp_results[svgp_results < -0.3] = -0.3
Z = svgp_results.reshape(_NUM_LONGS, _NUM_LATS)

fig = plt.figure(figsize=(10, 5))
ax = plt.subplot(projection=ccrs.Orthographic(-260, -20))
crs = ccrs.RotatedPole(pole_longitude=180)
CS = plt.contourf(X, Y, Z.T, levels=50, transform=crs, cmap='coolwarm', vmin=-0.3, vmax=0.3)
ax.gridlines()
ax.add_feature(cfeature.BORDERS, zorder=11, linestyle=':')
ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black')
ax.coastlines()
ax.set_global()
ax.set_title("Spherical SVGP", fontsize=20)

plt.tight_layout()
plt.savefig('figures/sla_spherical_svgp_predictions.pdf', bbox_inches='tight')
# %%