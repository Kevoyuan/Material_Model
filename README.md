# Material_Model — LS-DYNA Material Model Calibration

Research code for material characterization and calibration of anisotropic sheet-metal models using **experimental strain data**, **LS-DYNA finite-element simulations**, **YLD2000-2d yield-function analysis**, and **neural-network-assisted parameter inference**.

This repository was developed as part of a master's thesis. The workflow connects experimental measurements with numerical simulations to support material-model identification: generate candidate YLD2000-2d parameters, create and run LS-DYNA models, post-process strain and displacement results, construct machine-learning datasets, and evaluate inferred material parameters.

## Overview

The project focuses on a biaxial tensile-test workflow for material-model calibration. Its main building blocks are:

- **Experimental data processing** — align and process measured strain distributions.
- **YLD2000-2d parameter identification** — calculate and visualize anisotropic yield-function parameters.
- **LS-DYNA model generation and automation** — create keyword files and execute batches of FE simulations.
- **LS-PrePost post-processing** — extract displacement, strain, thickness, and geometric information from simulation results.
- **Dataset generation** — convert simulation and experimental results into inputs and targets for data-driven calibration.
- **Neural-network-assisted inference** — run NNabla / Neural Network Console inference workflows for material parameters.

## Workflow

A typical dataset-generation and calibration workflow is organized as the following stages:

| Stage | Script | Purpose |
| --- | --- | --- |
| P1 | `make_dataset/scripts/P1_cre_para_csv_3.py` | Generate material-property samples and identify corresponding YLD2000-2d parameters. |
| P2 | `make_dataset/scripts/P2_cre_key_files.py` | Create LS-DYNA keyword files for the generated parameter sets. |
| P3 | `make_dataset/scripts/P3_cre_outp_standardization*.py` | Standardize material parameters used as dataset outputs. |
| P4 | `make_dataset/scripts/P4_dyna_lsprepost_run.py` | Run LS-DYNA analyses and LS-PrePost post-processing in batch. |
| P5 | `make_dataset/scripts/P5_cre_inp_10.py` | Extract and assemble simulation results into model inputs. |
| P6 | `make_dataset/scripts/P6_*.py` | Evaluate yield-function / parameter errors and related quantities. |
| P7 | `make_dataset/scripts/P7_edit_ex_modify_data3.py` | Convert experimental data into the format required by the inference workflow. |
| P8 | `make_dataset/scripts/P8_nnabla_run2.py` | Run NNabla CLI forward inference and collect inferred parameters. |

`make_dataset/main_before_CNN.py` contains an example orchestration of several of these stages and shows how the LS-DYNA and LS-PrePost executable paths are configured.

## Project Structure

```text
.
├── README.md
├── LICENSE
├── tensile_test.py              # Experimental tensile-test data processing
├── FEM_model_modify.py          # FEM result extraction and model/post-file processing
├── fit_tri.ipynb                # Notebook for exploratory fitting / analysis
│
├── experiment_data/             # Experimental datasets
├── analysis/                    # Analysis scripts
│
├── make_dataset/
│   ├── main_before_CNN.py       # Example dataset/simulation orchestration
│   └── scripts/                 # P1–P8 dataset-generation and inference workflow
│
├── ML/
│   └── nnc/
│       └── EX2NN.py             # Experimental/simulation strain preprocessing for NNC input
│
├── yld2000/                     # YLD2000-2d parameter calculation and visualization
├── YLD_2d_Investigation/        # Additional YLD2000-2d investigations and yield-curve tools
└── utils/                       # Helper scripts and utilities
```

## Requirements

### External software

Some workflows require software that is not included in this repository:

- **LS-DYNA** — finite-element solver.
- **LS-PrePost** — LS-DYNA post-processing.
- **NNabla CLI / Neural Network Console** — required for the P8 inference workflow.

### Python

The scripts are written for Python 3 and use packages including:

```text
numpy
pandas
matplotlib
scipy
sympy
tqdm
```

Jupyter is useful for working with `fit_tri.ipynb`.

> **Note:** the repository currently does not contain a pinned `requirements.txt`. Install the packages required by the scripts you plan to run.

## Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/Kevoyuan/Material_Model.git
cd Material_Model

python -m venv .venv
```

Activate the environment:

```bash
# Windows
.\.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Install the main Python dependencies:

```bash
pip install numpy pandas matplotlib scipy sympy tqdm jupyter
```

## Usage

This repository contains research scripts rather than a packaged command-line application. Before running the simulation or inference workflows, review the file paths and solver settings inside the relevant scripts.

### Experimental data analysis

`tensile_test.py` contains utilities for reading tensile-test CSV data, calibrating specimen coordinates, processing strain distributions, and visualizing results.

```bash
python tensile_test.py
```

### FEM processing

`FEM_model_modify.py` combines FEM result extraction, strain calculations, YLD2000-2d utilities, and post-processing helpers. It is intended to be used together with the supporting analysis/helper modules and generated LS-DYNA result folders.

### Dataset generation

The main dataset workflow lives in `make_dataset/scripts/`. The P1–P8 scripts are designed as sequential research-processing stages rather than independent generic CLI tools.

`make_dataset/main_before_CNN.py` demonstrates the simulation-stage configuration, including CPU count, LS-DYNA solver location, LS-PrePost location, and post-processing command files.

### NNabla inference

`make_dataset/scripts/P8_nnabla_run2.py` invokes:

```text
nnabla_cli forward
```

to run trained `.nnp` models on prepared experimental input data and collect the inferred material parameters.

`ML/nnc/EX2NN.py` is a preprocessing helper that reduces simulation strain data and updates CSV paths used by the Neural Network Console workflow; it is not itself a neural-network training script.

## Configuration Notes

Several scripts were developed for a specific Windows research environment and therefore contain machine-specific paths such as LS-DYNA executables, LS-PrePost executables, `.nnp` files, CSV datasets, and output directories.

Before reproducing a workflow on another machine, check and update these paths, especially in:

- `make_dataset/main_before_CNN.py`
- `make_dataset/scripts/P4_dyna_lsprepost_run.py`
- `make_dataset/scripts/P8_nnabla_run2.py`
- `ML/nnc/EX2NN.py`

For reproducible studies, it is recommended to keep solver versions, material-model settings, parameter ranges, and preprocessing choices fixed and documented together with the generated dataset.

## Key Components

### YLD2000-2d

The `yld2000/` package contains tools for YLD2000-2d parameter calculation and yield-surface visualization, including:

- `YLD2000_2d_realM_EN.py`
- `Draw_YieldFunction_M_for_real.py`
- `plot_mult_yld.py`

### Experimental Data Processing

The repository includes workflows for processing tensile/biaxial-test data, strain distributions, specimen-coordinate calibration, and comparison between experiments and FEM results.

### Simulation-to-ML Pipeline

The `make_dataset/` workflow links material parameters to LS-DYNA simulation results and prepares the resulting strain/response data for data-driven material calibration.

## Author

- [Zhengxuan Yuan (@Kevoyuan)](https://github.com/Kevoyuan)

## Contributing

Contributions, bug reports, and improvements are welcome. For larger changes, please open an issue or pull request describing the proposed modification and the workflow it affects.

## License

This project is licensed under the [MIT License](LICENSE).
