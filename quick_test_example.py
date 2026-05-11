"""Quick smoke test for ChemoGAN modules.

Run:
    python quick_test_example.py
"""

from __future__ import annotations

import importlib
import sys

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
    "todos",
]


def main() -> int:
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
