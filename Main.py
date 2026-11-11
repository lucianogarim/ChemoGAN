"""Entry point kept for backward compatibility (revision 1).

The first-submission workflow (single run, PVR column-name bug, random point-level
validation split) has been replaced by run_experiments.py. This wrapper reproduces the
full revision-1 protocol for one target:

    python Main.py DWSI        # or DWCA

and is equivalent to

    python run_experiments.py --target DWSI --models chemogan gan_uncon chemogan_soft tvae ddpm
"""
import sys
import run_experiments

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "DWSI"
    sys.argv = [sys.argv[0], "--target", target, "--models", "chemogan", "gan_uncon",
                "chemogan_soft", "tvae", "ddpm", "--gan_epochs", "300", "--ddpm_epochs", "200"]
    run_experiments.main()
