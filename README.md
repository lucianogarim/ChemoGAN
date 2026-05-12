# ChemoGAN

ChemoGAN is a research-oriented repository for generating and evaluating synthetic petrophysical well-log profiles using deep-learning models.

The repository includes GAN-, diffusion-, and transformer-based generative workflows, as well as preprocessing, visualization, and XGBoost-based regression utilities. The quick-test workflow focuses on the synthetic generation of petrophysical profiles, and the downstream regression task is configured for the **DWSI** profile.

## Important Project Notes

* **Main entry point:** `Main.py` is the primary experiment script.
* **Quick smoke test:** `quick_test_example.py` checks whether the required project modules can be imported successfully.
* **Current regression target:** the XGBoost regression task is configured for **DWSI**.
* **Data location:** the LAS files are included in the local `wells/` folder.

## Project Structure

* `Main.py`: main experiment script; run this file for the full workflow.
* `quick_test_example.py`: lightweight import-based smoke test for validating the environment.
* `gan_lib.py` / `gan_trainer.py`: GAN architecture, physics-informed penalties, and training loop.
* `diffusion_lib.py` / `diffusion_trainer.py`: diffusion-model denoiser architecture and training/sampling workflow.
* `transformer_lib.py` / `transformer_trainer.py`: Transformer VAE architecture and training workflow.
* `xgb.py`: XGBoost regression workflow used to evaluate the DWSI prediction task.
* `pre_processamento.py`: LAS loading, depth filtering, outlier removal, scaling, and sequence construction utilities.
* `plots.py`: visualization and diagnostic plotting utilities.
* `wells/`: expected local folder for the `.las` well-log files.

## Hardware Requirements

Minimum for quick tests and small experiments:

* CPU: 4 cores
* RAM: 8 GB
* Storage: 5 GB free disk space
* GPU: optional

Recommended for model training:

* CPU: 8+ cores
* RAM: 16–32 GB
* Storage: 20+ GB SSD
* GPU: NVIDIA GPU with CUDA support and 8+ GB VRAM

The full training workflow can be computationally intensive, especially when running GAN, Transformer, and diffusion models with many epochs.

## Program Language

* **Primary language:** Python 3

## Software Requirements

* Python **3.10+**
* `pip` or `conda` for dependency management
* Recommended Python packages:

  * `tensorflow`
  * `numpy`
  * `pandas`
  * `scipy`
  * `scikit-learn`
  * `xgboost`
  * `matplotlib`
  * `seaborn`
  * `tqdm`
  * `lasio`
  * `openpyxl`

Optional but recommended:

* CUDA Toolkit and compatible NVIDIA drivers for GPU training
* Jupyter Notebook/Lab for interactive experiments

> Note: this project uses TensorFlow/Keras for the deep-learning models. PyTorch is not required unless you add separate Torch-based experiments.

## Setup

1. Clone this repository:

   ```bash
   git clone <your-repository-url>
   cd ChemoGAN
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   # .venv\Scripts\activate    # Windows PowerShell
   ```

3. Install dependencies:

   ```bash
   pip install -U pip
   pip install tensorflow numpy pandas scipy scikit-learn xgboost matplotlib seaborn tqdm lasio openpyxl
   ```

4. Add the dataset.

   The simplest setup is to place the `.las` files directly inside the `wells/` folder:

   ```text
   ChemoGAN/
   ├── wells/
   │   ├── 7-SPH-3-SPS_BASE.las
   │   ├── 7-SPH-20D-SPS_BASE.las
   │   └── ...
   ├── Main.py
   └── ...
   ```

   The current script uses:

   ```python
   data_dir = "wells"
   ```

   If you prefer to keep the data elsewhere, update the `data_dir` variable in `Main.py` or adapt the script to read from an environment variable such as `CHEMOGAN_DATA_DIR`.

## Quick Smoke Test

Before running the full workflow, run:

```bash
python quick_test_example.py
```

This script checks whether the required modules can be imported successfully. It does not train models, generate synthetic data, or run the DWSI regression task. It is intended as a lightweight environment validation step.

The broader quick-test workflow focuses on generating synthetic petrophysical well-log profiles. The downstream regression task uses **DWSI** as the target profile, with the remaining selected petrophysical logs used as predictors.

## Usage

Run the main experiment script:

```bash
python Main.py
```

The `Main.py` workflow performs the following high-level steps:

1. Loads LAS well-log files from `wells/`.
2. Applies depth filtering and outlier removal.
3. Selects blind wells for validation.
4. Scales the selected petrophysical logs.
5. Creates sequential windows for model training.
6. Trains and evaluates the configured generative models.
7. Generates synthetic petrophysical profiles.
8. Uses XGBoost regression to evaluate the DWSI prediction task.
9. Saves summary outputs and comparison tables.

Alternative modules are usually imported by `Main.py`, but they can also be inspected or extended individually:

```bash
python gan_trainer.py
python diffusion_trainer.py
python transformer_trainer.py
python xgb.py
```

> Tip: use the quick smoke test first, then run a reduced number of epochs before launching long training jobs.

## Outputs

Depending on the selected workflow, the repository may generate:

* training-convergence plots such as `chemogan_convergence.png`;
* blind-well comparison files such as `comparacao_pocos_cegos.csv` or `.xlsx` files;
* summary tables comparing generative models and regression performance;
* diagnostic plots for distributions, correlations, PCA projections, and well profiles.

## Reproducibility Notes

* Keep the preprocessing steps synchronized with model input expectations.
* Keep the same feature order between preprocessing, model training, synthetic generation, and regression.
* Track random seeds, hyperparameters, dataset version, and training epochs.
* Save generated plots and output tables for each experiment run.
* Use separate output folders per model type when comparing GAN, Transformer, and diffusion results.
* For journal submission, archive the exact dataset version and code release used in the manuscript.

## Contributing

1. Create a feature branch.
2. Commit changes with clear messages.
3. Open a pull request describing the motivation, implementation, and validation.

## License

Add a license file, such as MIT or Apache-2.0, to define usage terms.
