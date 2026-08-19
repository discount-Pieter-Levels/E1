"""
run_e1.py — ENTRY POINT for the E1 depth-scaling sweep.

This is the file you launch:  python3 -u run_e1.py

It glues three things together:
  1. depth_eggroll.py     — the depth-parameterized DepthCSNN + EGGROLL machinery
  2. e1_harness.py        — run_depth_sweep(), which drives the depth x seed grid
  3. YOUR main training script — provides run_eggroll_csnn() and the config constants

>>> PREREQUISITE (must be done first, see E1_INTEGRATION.md) <<<
Your main training script's run_eggroll_csnn() MUST be edited to:
  (a) accept a prebuilt `net=None` argument (use it if given, else build CSNN())
  (b) use make_all_factors_depth() + depth_forward_subbatch() from depth_eggroll
  (c) apply gradients in a loop over net.depth conv blocks (not hardcoded conv1/2/3)
Until that edit is done, this script will import fine but the runs won't use the
depth architecture correctly.

Change MAIN_MODULE below to the actual filename (without .py) of your training
script on the pod.
"""
import os, torch

# ---- point this at YOUR training script's module name (filename without .py) ----
MAIN_MODULE = os.environ.get("MAIN_MODULE", "CSNN_sota_matrix")

from depth_eggroll import DepthCSNN
from e1_harness import run_depth_sweep

# import your training infra + config from your main script
_main = __import__(MAIN_MODULE)
run_eggroll_csnn = _main.run_eggroll_csnn
DEVICE      = _main.DEVICE
C           = _main.C
H           = _main.H
NUM_CLASSES = _main.NUM_CLASSES
NUM_STEPS   = _main.NUM_STEPS
BETA        = _main.BETA
SIGMA0      = getattr(_main, "SIGMA0", getattr(_main, "SIGMA_0", None))
SIGMA_DECAY = _main.SIGMA_DECAY
SIGMA_MIN   = _main.SIGMA_MIN
LR_ES       = _main.LR_ES
EVAL_BATCHES_K = _main.EVAL_BATCHES_K
OUT_DIR     = getattr(_main, "OUT_DIR", "./results/E1")

# ---- experiment knobs (all overridable via env for parallel launches) ----
DEPTHS      = [int(x) for x in os.environ.get("DEPTHS", "2 3 4 5 6").split()]
SEEDS       = [int(x) for x in os.environ.get("SEEDS",  "0 1 2").split()]
POPULATION  = int(os.environ.get("POPULATION", "256"))   # HELD FIXED across depths
RANK        = int(os.environ.get("RANK", "16"))
GENERATIONS = int(os.environ.get("GENERATIONS", "1000"))
DATASET_TAG = os.environ.get("DATASET", "nmnist")

os.makedirs(OUT_DIR, exist_ok=True)


def build_net(depth, seed):
    torch.manual_seed(seed)
    return DepthCSNN(depth, in_ch=C, in_hw=H, num_classes=NUM_CLASSES,
                     num_steps=NUM_STEPS, beta=BETA).to(DEVICE)


def train_fn(net, generations, population, rank, lr, seed):
    # early stopping OFF (huge patience) so every seed shares a fixed x-axis and
    # curves are directly averageable across seeds.
    return run_eggroll_csnn(
        generations=generations, population=population, rank=rank,
        sigma0=SIGMA0, sigma_decay=SIGMA_DECAY, sigma_min=SIGMA_MIN,
        lr=lr, eval_batches_k=EVAL_BATCHES_K, seed=seed,
        net=net, patience=10**9)


if __name__ == "__main__":
    print(f"### E1 sweep | dataset={DATASET_TAG} depths={DEPTHS} seeds={SEEDS} "
          f"pop={POPULATION} rank={RANK} gens={GENERATIONS} main={MAIN_MODULE}")
    run_depth_sweep(
        depths=DEPTHS, seeds=SEEDS,
        build_net=build_net, train_fn=train_fn,
        generations=GENERATIONS, population=POPULATION,
        rank=RANK, lr=LR_ES,
        chance=100.0 / NUM_CLASSES,
        out_dir=OUT_DIR, dataset_tag=DATASET_TAG)
