#!/bin/bash

#SBATCH --job-name=pm_recon
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=300GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=iaifi_lab
#SBATCH -p iaifi_gpu_priority
#SBATCH --array=0-8

cd /n/home11/ccuestalazaro/CosmoTransfer/ctransfer

commands=(
  "python train.py --n_baseline 10_000 --n_shots 100 --few_shot_cosmological_parameters 'M_nu' --freeze_summarizer --run_name 'vit_fine_tune_M_nu_freeze_summarizer' --summarizer vit"
  "python train.py --n_baseline 10_000 --n_shots 100 --few_shot_cosmological_parameters 'w' --freeze_summarizer --run_name 'vit_fine_tune_w_freeze_summarizer' --summarizer vit"
  "python train.py --n_baseline 10_000 --n_shots 100 --few_shot_cosmological_parameters 'M_nu' --run_name 'vit_fine_tune_M_nu' --summarizer vit"
  "python train.py --n_baseline 10_000 --n_shots 100 --few_shot_cosmological_parameters 'w' --run_name 'vit_fine_tune_w' --summarizer vit"
  "python train.py --n_baseline 100 --n_shots 0 --cosmological_parameters 'Omega_m' 'Omega_b' 'h' 'n_s' 'sigma_8' 'M_nu' --run_name 'vit_from_scratch_small_M_nu' --summarizer vit"
  "python train.py --n_baseline 100 --n_shots 0 --cosmological_parameters 'Omega_m' 'Omega_b' 'h' 'n_s' 'sigma_8' 'w' --run_name 'vit_from_scratch_small_w' --summarizer vit"
  "python train.py --n_baseline 2_000 --n_shots 0 --cosmological_parameters 'Omega_m' 'Omega_b' 'h' 'n_s' 'sigma_8' 'M_nu' --run_name 'vit_from_scratch_all_M_nu' --summarizer vit"
  "python train.py --n_baseline 2_000 --n_shots 0 --cosmological_parameters 'Omega_m' 'Omega_b' 'h' 'n_s' 'sigma_8' 'w' --run_name 'vit_from_scratch_all_w' --summarizer vit"
)
# commands=(
#   "python train.py --n_baseline 10_000 --n_shots 100 --few_shot_cosmological_parameters 'M_nu' --freeze_summarizer --run_name 'fine_tune_M_nu_freeze_summarizer'"
#   "python train.py --n_baseline 10_000 --n_shots 100 --few_shot_cosmological_parameters 'w' --freeze_summarizer --run_name 'fine_tune_w_freeze_summarizer'"
#   "python train.py --n_baseline 10_000 --n_shots 100 --few_shot_cosmological_parameters 'M_nu' --run_name 'fine_tune_M_nu'"
#   "python train.py --n_baseline 10_000 --n_shots 100 --few_shot_cosmological_parameters 'w' --run_name 'fine_tune_w'"
#   "python train.py --n_baseline 100 --n_shots 0 --cosmological_parameters 'Omega_m' 'Omega_b' 'h' 'n_s' 'sigma_8' 'M_nu' --run_name 'from_scratch_small_M_nu'"
#   "python train.py --n_baseline 100 --n_shots 0 --cosmological_parameters 'Omega_m' 'Omega_b' 'h' 'n_s' 'sigma_8' 'w' --run_name 'from_scratch_small_w'"
#   "python train.py --n_baseline 2_000 --n_shots 0 --cosmological_parameters 'Omega_m' 'Omega_b' 'h' 'n_s' 'sigma_8' 'M_nu' --run_name 'from_scratch_all_M_nu'"
#   "python train.py --n_baseline 2_000 --n_shots 0 --cosmological_parameters 'Omega_m' 'Omega_b' 'h' 'n_s' 'sigma_8' 'w' --run_name 'from_scratch_all_w'"
# )

eval ${commands[$SLURM_ARRAY_TASK_ID]}
