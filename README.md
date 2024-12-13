# Deep Random Features

<!---
### Repository containing the official PyTorch implementation of the Deep Random Features for Scalable Interpolation of Spatiotemporal Data.

---
-->

## Overview


**Deep Random Features (DRF)** is a Bayesian deep learning framework designed for scalable interpolation of spatiotemporal data. By leveraging random feature expansions, DRF layers incorporate the inductive bias of stationary Gaussian Processes (GPs), supporting applications both on the **plane** and on the **sphere**. 

DRF is especially well-suited for interpolating diverse types of spatiotemporal data, such as **local** and **global satellite readings**, and offers a flexible approach for modeling complex data patterns. The primary aim of this repository is to:
- Ensure **reproducibility** of the experiments presented in the paper.
- Provide an **easy-to-use implementation** of DRF, making it accessible to researchers as a specialised deep neural network.

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

*Figure 1: Satellite Measurements alongside DRF Predictions and Uncertainties.*

## Installation and Usage
The implementation is purely in Python, and this repository includes example scripts and configuration files to help you get started with DRF.

1. **Clone the Repository**

   ```bash
   git clone git@github.com:totony4real/DeepRandomFeatures.git
   cd DeepRandomFeatures

2. **Create a Virtual Environment**

    You can use conda commands or

    ```bash
    python3 -m venv venv
    source venv/bin/activate

3. Install Required Packages
    ```bash
    pip install -r requirements.txt    

4. Installs the DRF package from the locally cloned directory in an editable mode
    ```bash
    pip install -e ./  

---

## Experiments

### Data Access
The datasets used for these experiments can be downloaded from the following Google Drive link:
[Download Data](https://drive.google.com/drive/folders/17rwMtEc5vwRKEjNolreUBL2Yk4OSTvr4?usp=sharing)

The Google Drive folder contains the following files:

1. **exp1.zip**  
   - Used in **Experiment 1**.  
   - Configuration file reference: `zip_path` in `example_config1.yaml`.  
   - Example path: `'path/to/exp1.zip'`.

2. **exp2.pkl**  
   - Used in **Experiment 2**.  
   - Configuration file reference: `obs_datapath` in `example_config2.yaml`.  
   - Example path: `'path/to/exp2.pkl'`.

3. **exp2_test_loc.csv**  
   - Used in **Experiment 2**.  
   - Configuration file reference: `test_data_path` in `example_config2.yaml`.  
   - Example path: `'path/to/exp2_test_loc.csv'`.

4. **exp3.csv**  
   - Used in **Experiment 3**.  
   - Configuration file reference: `file_path` in `example_config3.yaml`.  
   - Example path: `'path/to/exp3.csv'`.

5. **exp3_test_data.pt**  
   - Used in **Experiment 3**.  
   - Configuration file reference: `test_data_path` in `example_config3.yaml`.  
   - Example path: `'path/to/exp3_test_data.pt'`.

Ensure that the files are placed in the appropriate paths as referenced in the respective configuration files before running the experiments.

---
### Experiment 1: Synthetic MSS Data
This experiment focuses on interpolating synthetic **Mean Sea Surface (MSS)** data using DRF. The synthetic data is used to evaluate DRF's performance in capturing underlying spatial patterns on a local scale, offering a controlled environment to test its interpolation accuracy.

**To run Experiment 1:**
```bash
python examples/experiment_1.py --config configs/example_config_exp1.yaml
``` 

### Experiment 2: ABC Freeboard Data
In this experiment, we use **ABC Freeboard data** to test DRF's ability to interpolate satellite-derived freeboard measurements. This data helps evaluate DRF's effectiveness in real-life scenarios where the data is noisy.

**To run Experiment 2:**
```bash
python examples/experiment_2.py --config configs/example_config_exp2.yaml
```

### Experiment 3: SLA Data
This experiment applies DRF to **Sea Level Anomaly (SLA)** data, assessing its capability to generalise and interpolate satellite measurements over the globe. The SLA data provides a global scale challenge for DRF, testing both its spatial and temporal predictive power in a more data-intensive setting.

**To run Experiment 3:**
```bash
python examples/experiment_3.py --config configs/example_config_exp3.yaml
```
---
### Results Plotting

#### Experiment 1 and Experiment 2
For **Experiment 1** and **Experiment 2**, the repository includes plotting notebooks located in the `plotting` folder. These scripts allow users to visualise the results obtained from the experiments, including comparisons with ground truth data where applicable.

To use these plotting scripts:
1. Navigate to the `plotting` folder in the repository.
2. Run the relevant notebook for the corresponding experiment, ensuring you provide any required paths or configuration details.

#### Experiment 3
For **Experiment 3**, the plotting is automated. The experiment script saves the generated plots directly to the paths specified in the configuration file under the `results` section. Users can define the desired save locations by editing the corresponding fields in `configs/example_config_exp3.yaml`.

---

## Citation
If you use this repository for your research, please cite our paper:
