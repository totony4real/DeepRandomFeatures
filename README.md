<h1>Deep Random Features</h1>

<h3>Repository containing the official PyTorch implementation of the Deep Random Features for Scalable Interpolation of Spatiotemporal Data. </h3>

## Overview
This repo contains the official PyTorch implementation of the Deep Random Features (DRF) model as introduced in ....(here insert link to the paper)
DRF is a Bayesian deep learning framework, whose layers share the inductive bias of stationary Gaussian Processes on both the plane and sphere via random feature expansions. We expect DRF to facilitate the interpolation of various types of spatiotemporal data (e.g. local and global satellite readings). The primary goal of this library is to provide researchers with reproducibility of the experiment and an easy to use implementation of DRF as a specialised deep neural networks.

<div style="display: flex; align-items: center; justify-content: center;">
    <div>
        <img src="images/sla_satellite_measurements.png" alt="Satellite Measurements" width="500" />
    </div>
    <div style="display: flex; flex-direction: column; margin-left: 10px;">
        <img src="images/sla_drf_predictions.png" alt="DRF Predictions" width="240" style="margin-bottom: 10px;" />
        <img src="images/sla_drf_uncertainties.png" alt="DRF Uncertainties" width="240" />
    </div>
</div>



