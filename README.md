# ChemoGAN

ChemoGAN is a research-oriented repository for training and evaluating deep-learning models for chemical data generation and analysis.  
The project contains multiple training pipelines (GAN, diffusion, transformer, and XGBoost-based workflows) plus preprocessing and plotting utilities.

## Important Project Notes

- **Main entry point:** `todos.py` is the main file and should be run as the primary experiment script.
- **Current experiment setup:** The experiment is currently configured for **DWSI**.
- **Dataset access:** The data used in this project is confidential. If you need access, please contact **lucianogarim@gmail.com**.

## Project Structure

- `todos.py`: **Main script** (run this file first).
- `gan_lib.py` / `gan_trainer.py`: GAN model components and training loop.
- `diffusion_lib.py` / `diffusion_trainer.py`: Diffusion model components and training loop.
- `transformer_lib.py` / `transformer_trainer.py`: Transformer model components and training loop.
- `xgb.py`: Gradient-boosting baseline/utility scripts.
- `pre_processamento.py`: Data preprocessing utilities.
- `plots.py`: Visualization and analysis plots.

## Hardware Requirements

Minimum (small experiments):
- CPU: 4 cores
- RAM: 8 GB
- Storage: 5 GB free disk space
- GPU: Optional (NVIDIA GPU with 4+ GB VRAM recommended for faster training)

Recommended (model training):
- CPU: 8+ cores
- RAM: 16–32 GB
- Storage: 20+ GB SSD
- GPU: NVIDIA GPU with CUDA support and 8+ GB VRAM

## Program Language

- **Primary language:** Python 3

## Software Required

- Python **3.10+**
- `pip` (or `conda`) for dependency management
- Recommended Python packages:
  - `torch`
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `xgboost`
  - `matplotlib`
  - `seaborn`
  - `tqdm`
- Optional but recommended:
  - CUDA Toolkit + compatible NVIDIA drivers (for GPU training)
  - Jupyter Notebook/Lab (for interactive experiments)

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
   # .venv\Scripts\activate      # Windows (PowerShell)
   ```

3. Install dependencies:
   ```bash
   pip install -U pip
   pip install torch numpy pandas scikit-learn xgboost matplotlib seaborn tqdm
   ```

## Usage

Run the main script:

```bash
python todos.py
```

Alternative training scripts (if needed):

```bash
python gan_trainer.py
python diffusion_trainer.py
python transformer_trainer.py
python xgb.py
```

> Tip: Start with a smaller dataset subset to validate your environment before launching long training jobs.

## Documentation Notes

- Keep preprocessing steps in sync with model input expectations.
- Track random seeds and hyperparameters for reproducibility.
- Save checkpoints and generated plots for each experiment run.
- Use separate output folders per model type to compare results cleanly.

## Contributing

1. Create a feature branch.
2. Commit your changes with clear messages.
3. Open a pull request describing motivation, implementation, and validation.

## License

Add your preferred license file (e.g., MIT, Apache-2.0) to define usage terms.
