<h1>Deep Random Features</h1>

<h3>Repository containing the official PyTorch implementation of the Deep Random Features for Scalable Interpolation of Spatiotemporal Data. </h3>

## Overview
This repo contains the official PyTorch implementation of the Deep Random Features (DRF) model as introduced in ....(here insert link to the paper)
DRF is a Bayesian deep learning framework, whose layers share the inductive bias of stationary Gaussian Processes on both the plane and sphere via random feature expansions. We expect DRF to facilitate the interpolation of various types of spatiotemporal data (e.g. local and global satellite readings). The primary goal of this library is to provide researchers with reproducibility of the experiment and an easy to use implementation of DRF as a specialised deep neural networks.

<table>
  <tr>
    <td rowspan="2" style="vertical-align: top;">
      <img src="images/sla_satellite_measurements.png" alt="Satellite Measurements" width="400" />
    </td>
    <td style="padding-left: 10px;">
      <img src="images/sla_drf_predictions.png" alt="DRF Predictions" width="200" />
    </td>
  </tr>
  <tr>
    <td style="padding-left: 10px;">
      <img src="images/sla_drf_uncertainties.png" alt="DRF Uncertainties" width="200" />
    </td>
  </tr>
</table>




