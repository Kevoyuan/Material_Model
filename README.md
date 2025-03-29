# LSDYNA Material Modeling Project

Advanced framework for biaxial tensile test simulations and material model development using LS-DYNA, machine learning, and numerical analysis.

## Features
- 🧪 Biaxial tensile test specimen modeling
- ⚙️ FEM analysis automation with LS-DYNA
- 🤖 Machine learning integration for material parameter calibration
- 📊 Experimental data processing pipeline
- 📈 Yield surface visualization and analysis
- 🔄 Automated batch processing workflows

## Requirements
- Python 3.8+
- NumPy
- pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook (for ML analysis)

## Quick Start
```bash
pip install numpy pandas matplotlib scikit-learn

# Run tensile test analysis
python tensile_test.py

# Generate FEM models
python FEM_model_modify.py

# Start ML training
cd ML/nnc
python EX2NN.py
```

## Project Structure
```
├── ML/ - Machine learning models and training pipelines
├── make_dataset/ - Data generation and preprocessing
├── yld2000/ - Yield function analysis and visualization
├── experiment_data/ - Raw experimental datasets
├── FEM_model_modify.py - Finite Element Model generator
├── tensile_test.py - Core tensile test analysis module
└── Modify_postfile.py - LS-DYNA result postprocessor
```

## Documentation

### Dataset Generation Pipeline
**Workflow Stages**:
1. `P1_cre_para_csv_3.py` - Generates random material parameters using symbolic math
2. `P2_cre_key_files.py` - Creates LS-DYNA keyword files from parameters
3. `P3_cre_outp_*.py` - Standardizes FEM simulation outputs
4. `P4_dyna_*.py` - Automated LS-DYNA batch processing
5. `P5_cre_inp_10.py` - Prepares ML-ready input tensors
6. `P6_*.py` - Calculates yield surface errors
7. `P7_edit_ex_modify_data3.py` - Validates experimental data formats and converts to ML-compatible structures
8. `P8_nnabla_run2.py` - Neural network training execution with hyperparameter configuration

**Key Features**:
- 🧩 Parameter space exploration with sympy
- ⚡ Parallel LS-DYNA execution
- 📈 Strain data normalization pipelines
- 🔄 ML dataset versioning support
- 🔍 Experimental data validation checks
- 🧠 Neural network configuration management

[Material Calibration Tutorial](ML/nnc/EX2NN.py) | [FEM Guide](FEM_model_modify.py)

### Experimental Data Processing
- Tensile/bulge test analysis
- Strain distribution visualization
- Experimental/FEM data correlation

### Machine Learning Integration
- Dataset split (train/val/test)
- Input feature engineering
- Output normalization layers
- Experimental data validation and preprocessing (P7)
- Neural network training workflows with hyperparameter configuration (P8)

## Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License
[MIT](LICENSE)
