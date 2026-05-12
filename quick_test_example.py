"""Quick smoke test for ChemoGAN modules.

This script performs a lightweight import test for the main ChemoGAN modules.
It is intended to verify that the project environment is correctly configured
and that all required Python files can be imported without errors.

The broader quick test associated with this repository focuses on the synthetic
generation of petrophysical well-log profiles. In the downstream regression
task, the target profile is DWSI, while the remaining selected petrophysical
logs are used as predictors.

This script does not train the models or generate synthetic data directly.
Instead, it checks whether the codebase is ready to run the full quick test
pipeline.

Run:
    python quick_test_example.py
"""

from __future__ import annotations

import importlib
import sys


# List of project modules required by the ChemoGAN quick test workflow.
#
# These modules cover:
# - GAN-based synthetic profile generation;
# - Transformer-based synthetic profile generation;
# - Diffusion-based synthetic profile generation;
# - LAS preprocessing and sequence construction;
# - plotting utilities;
# - XGBoost regression for the DWSI target profile;
# - the main experiment script.
MODULES = [
    "gan_lib",
    "gan_trainer",
    "diffusion_lib",
    "diffusion_trainer",
    "transformer_lib",
    "transformer_trainer",
    "pre_processamento",
    "plots",
    "xgb",
    "Main",
]


def main() -> int:
    """Import all required modules and report whether the environment is valid.

    Returns
    -------
    int
        Exit code compatible with command-line execution:
        - 0 if all modules are imported successfully;
        - 1 if one or more imports fail.
    """
    failures = []

    for name in MODULES:
        try:
            importlib.import_module(name)
            print(f"[OK] import {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"[FAIL] import {name}: {exc}")

    if failures:
        print(f"\n{len(failures)} module(s) failed to import.")
        return 1

    print("\nQuick test finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())