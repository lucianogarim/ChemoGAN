#!/bin/sh
# Reproduces every experiment of the revised manuscript (CAGEO-D-26-00775, revision 1).
# CPU (2 cores): about 20 h in total; a GPU is much faster. Completed runs are skipped,
# so the script can be interrupted and restarted.
cd "$(dirname "$0")"
for T in DWSI DWCA; do
  # GAN variants, 5 seeds (ChemoGAN-H = chemogan, ChemoGAN-P = chemogan_soft);
  # gan_uncon also produces the volume-matched rejection set (gan_uncon_matched).
  python3 run_experiments.py --target $T --gan_epochs 300 --models chemogan chemogan_soft gan_uncon --seeds 0 1 2 3 4
  # Baselines, 3 seeds (tvae also produces tvae_prior from the same trained model)
  python3 run_experiments.py --target $T --ddpm_epochs 200 --tvae_epochs 150 --models tvae ddpm --seeds 0 1 2
done
python3 analysis.py > results/summary/analysis_stdout.txt
python3 diagnostics.py > results/summary/diagnostics_stdout.txt
python3 make_tables.py
python3 figures.py
