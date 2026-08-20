#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =============================================================================
# Evolutionary CSNN (EGGROLL) vs Surrogate-Gradient CSNN — N-MNIST
#
# Architecture (both models) — SOTA-oriented, 3 conv blocks + BNTT:
#   Input  (B, T, 2, 34, 34)
#   Conv2d(2, 32, k=3, pad=1)  + BN(t) + LIF   -> (B, T, 32, 34, 34)
#   AvgPool2d(2)                                -> (B, T, 32, 17, 17)
#   Conv2d(32, 64, k=3, pad=1) + BN(t) + LIF   -> (B, T, 64, 17, 17)
#   AvgPool2d(2)                                -> (B, T, 64, 8, 8)
#   Conv2d(64, 128, k=3, pad=1)+ BN(t) + LIF   -> (B, T, 128, 8, 8)
#   AvgPool2d(2)                                -> (B, T, 128, 4, 4)
#   Flatten                                     -> (B, T, 2048)
#   Linear(2048, 10) + leaky readout            -> (B, T, 10)
#
# EGGROLL extension to conv layers:
#   A conv layer is a linear map once you unfold the input into patches
#   (im2col). For input patches X_unfold and flattened weight W_flat:
#       conv_output = W_flat @ X_unfold        (then reshape to feature map)
#   so the same low-rank correction used for linear layers applies:
#       output = W_flat @ X_unfold + scale * A @ (B_lr.T @ X_unfold)
#   implemented via F.unfold so the linear-case machinery carries over.
#
# BNTT (Batch Norm Through Time) — added back, but deliberately AFFINE-FREE
# on the EGGROLL path:
#   - CSNN's bn1/bn2/bn3 use affine=False: pure (x-mean)/std standardisation
#     per timestep, with NO learnable gamma/beta. ES therefore has nothing
#     extra to perturb for BN, so none of the per-member-BN-factor gradient
#     machinery from the earlier attempt is needed -- that machinery is what
#     caused several of the earlier bugs and OOMs. This keeps the parameter
#     count, factor generation, and gradient reconstruction IDENTICAL in
#     structure to the no-BN version; BN is purely a normalisation step
#     inserted between conv and LIF.
#   - SurrogateCSNN's bn1/bn2/bn3 use affine=True (the BPTT literature
#     default) since backprop handles the extra learnable params for free.
#   - CRITICAL correctness point: BN must run in TRAIN mode during ES
#     population evaluation, or its running stats never update and an
#     affine=False BN with default running_mean=0/running_var=1 becomes a
#     literal no-op (output == input). evaluate_eggroll therefore now calls
#     base_net.train() instead of base_net.eval() -- see that cell.
#
# Bug fixes applied throughout:
#   - Centered rank normalisation, NOT z-score.
#   - Adam optimiser, not raw SGD on the gradient estimate.
#   - Fan-in scaled weight init: std = 0.3 / sqrt(fan_in), so deep layers
#     don't saturate their LIF neurons.
#   - PER-LAYER PERTURBATION SCALE: sigma is a RELATIVE perturbation
#     fraction. Each layer's actual perturbation std = sigma * (that
#     layer's init std), so every layer is perturbed by the same fraction
#     of its OWN weights, regardless of how different their absolute
#     scales are after fan-in init.
#   - eggroll_forward_subbatch's `data` arrives ALREADY tiled across the
#     population (shape P_sub*db), so state tensors (mem1/mem2/mem3) are
#     NOT multiplied by P_sub a second time -- that double-tiling was the
#     actual cause of the earlier OOM, not BN or conv3 themselves.
#   - Validation accuracy is evaluated every VAL_EVERY generations instead
#     of every single generation.
#   - Sigma decay computed from a target end-of-training sigma (now
#     overridden by the cosine warm-restart schedule in run_eggroll_csnn).
# =============================================================================

# !pip install snntorch tonic torch numpy matplotlib

import os, time, json, math
import numpy as np
import torch
from depth_eggroll import make_all_factors_depth, depth_forward_subbatch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader, random_split
import tonic
import tonic.transforms as tonic_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# =============================================================================
# CONFIG
# =============================================================================

BATCH_SIZE  = 64
DATA_PATH   = "./data/nmnist"
os.makedirs(DATA_PATH, exist_ok=True)

SENSOR_SIZE = tonic.datasets.NMNIST.sensor_size   # (34, 34, 2)
H, W, C     = SENSOR_SIZE
NUM_STEPS   = 20
NUM_CLASSES = 10
BETA        = 0.95

# Architecture dims — now THREE conv+pool stages.
CONV1_OUT, CONV1_K, CONV1_PAD = 32, 3, 1
CONV2_OUT, CONV2_K, CONV2_PAD = 64, 3, 1
CONV3_OUT, CONV3_K, CONV3_PAD = 128, 3, 1
POOL_K = 2

# Explicit spatial-dimension chain — computed ONCE here and reused everywhere
# (CSNN.forward, SurrogateCSNN.forward, eggroll_forward_subbatch) instead of
# being re-derived inline in multiple places, which is exactly the kind of
# duplication that produced the earlier FC_IN mismatch bugs.
#   conv uses stride=1 + 'same' padding, so conv never changes spatial size;
#   only AvgPool2d(2) does, via floor division.
H2, W2 = H  // POOL_K, W  // POOL_K   # 17, 17 -- after pool1
H3, W3 = H2 // POOL_K, W2 // POOL_K   # 8,  8  -- after pool2 (== conv3 in/out size)
H4, W4 = H3 // POOL_K, W3 // POOL_K   # 4,  4  -- after pool3
FC_IN  = CONV3_OUT * H4 * W4          # 128 * 4 * 4 = 2048

# Weight-init scale. Shared by BOTH the model init and the ES perturbation
# scale so that perturbations are a fixed fraction of each layer's weights.
BASE_STD = 0.3
INIT_STD_CONV1 = BASE_STD / (C         * CONV1_K * CONV1_K) ** 0.5  # fan_in 18
INIT_STD_CONV2 = BASE_STD / (CONV1_OUT * CONV2_K * CONV2_K) ** 0.5  # fan_in 288
INIT_STD_CONV3 = BASE_STD / (CONV2_OUT * CONV3_K * CONV3_K) ** 0.5  # fan_in 576
INIT_STD_FC    = BASE_STD / FC_IN ** 0.5                            # fan_in 2048

# ES / EGGROLL
GENERATIONS    = 3000
POPULATION     = 384
# Lowered from 16 -> 8: three conv layers plus per-timestep BN buffers means
# more live activation memory per sub-batch than the two-conv, no-BN version.
# Raise back toward 16 if your GPU has headroom once a run is stable.
POP_SUBBATCH   = 64
RANK           = 16
SIGMA0         = 0.05   # RELATIVE perturbation: 5% of each layer's weight scale
SIGMA_TARGET_END = 0.02          # target RELATIVE sigma at the final generation
SIGMA_DECAY    = (SIGMA_TARGET_END / SIGMA0) ** (1.0 / GENERATIONS)
SIGMA_MIN      = 0.02             # floor on the RELATIVE sigma (1%)
LR_ES          = 0.001
EVAL_BATCHES_K = 2
VAL_EVERY      = 20               # run a full val pass every N generations
SEED           = 0
PATIENCE = 250
MIN_DELTA = 0.1
T_0    = 600     # first restart cycle length in generations
T_MULT = 2       # each subsequent cycle is 2x longer

# Surrogate BPTT baseline
BPTT_EPOCHS = 15
LR_BPTT     = 1e-3

OUT_DIR = "./eggroll_csnn_nmnist_results"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"sigma_decay computed = {SIGMA_DECAY:.6f}  "
      f"(sigma_0={SIGMA0} -> sigma_end\u2248{SIGMA0 * SIGMA_DECAY**GENERATIONS:.4f} "
      f"after {GENERATIONS} generations)")
print(f"FC_IN = {FC_IN}  (spatial chain: {H}x{W} -> {H2}x{W2} -> {H3}x{W3} -> {H4}x{W4})")

# =============================================================================
# DATA
# =============================================================================
frame_transform = tonic_transforms.ToFrame(sensor_size=SENSOR_SIZE, n_time_bins=NUM_STEPS)

full_train  = tonic.datasets.NMNIST(save_to=DATA_PATH, train=True,  transform=frame_transform)
nmnist_test = tonic.datasets.NMNIST(save_to=DATA_PATH, train=False, transform=frame_transform)

train_size = int(0.9 * len(full_train))
val_size   = len(full_train) - train_size
nmnist_train, nmnist_val = random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(0)
)


# In[2]:


def collate_fn(batch):
    frames, labels = zip(*batch)
    frames = torch.stack([torch.from_numpy(f).float() for f in frames])
    labels = torch.tensor(labels, dtype=torch.long)
    return frames, labels   # frames: (B, T, 2, 34, 34)

train_loader = DataLoader(nmnist_train, batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=True,  collate_fn=collate_fn)
val_loader   = DataLoader(nmnist_val,   batch_size=BATCH_SIZE, shuffle=False,
                          drop_last=False, collate_fn=collate_fn)
test_loader  = DataLoader(nmnist_test,  batch_size=BATCH_SIZE, shuffle=False,
                          drop_last=False, collate_fn=collate_fn)

# In[3]:


def clamp_events(x):
    """Event frames can have counts > 1; clamp to binary spikes."""
    return x.clamp(0, 1)

# =============================================================================
# MODELS
# =============================================================================

# In[4]:


def fan_in_init_(m, base_std=0.3):
    """
    Scale per-weight std by 1/sqrt(fan_in) so pre-activation magnitude stays
    roughly constant across layers regardless of how many inputs they sum.
    Without this, deeper layers get much larger pre-activations than conv1,
    saturating their LIF neurons into firing every timestep regardless of
    input.
    """
    if isinstance(m, nn.Conv2d):
        fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
    elif isinstance(m, nn.Linear):
        fan_in = m.in_features
    else:
        raise ValueError(f"Unsupported module type: {type(m)}")
    nn.init.normal_(m.weight, mean=0.0, std=base_std / (fan_in ** 0.5))


class CSNN(nn.Module):
    """
    Plain (unperturbed) conv LIF network — the parameter container that ES
    trains. ES never calls .forward() during training (it goes through
    eggroll_forward_subbatch instead, perturbing copies of these same
    weights for the whole population); .forward() is only used by
    accuracy_on_loader for validation/test evaluation.

    BatchNorm here uses affine=False: pure per-timestep standardisation,
    with NO learnable gamma/beta. This means ES has nothing extra to
    perturb or compute a gradient for -- BN is just a normalisation step,
    not an additional set of trainable parameters on the ES path. The
    asymmetry with SurrogateCSNN (which uses affine=True) is deliberate:
    BPTT gets the full learnable BN for free via autograd, ES gets the
    normalisation benefit without the added factor/gradient machinery
    that caused earlier bugs.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(C, CONV1_OUT, CONV1_K, padding=CONV1_PAD)
        self.bn1   = nn.ModuleList([nn.BatchNorm2d(CONV1_OUT, affine=False)
                                     for _ in range(NUM_STEPS)])
        self.lif1  = snn.Leaky(beta=BETA, learn_beta=True, learn_threshold=True)
        self.pool1 = nn.AvgPool2d(POOL_K)

        self.conv2 = nn.Conv2d(CONV1_OUT, CONV2_OUT, CONV2_K, padding=CONV2_PAD)
        self.bn2   = nn.ModuleList([nn.BatchNorm2d(CONV2_OUT, affine=False)
                                     for _ in range(NUM_STEPS)])
        self.lif2  = snn.Leaky(beta=BETA, learn_beta=True, learn_threshold=True)
        self.pool2 = nn.AvgPool2d(POOL_K)

        self.conv3 = nn.Conv2d(CONV2_OUT, CONV3_OUT, CONV3_K, padding=CONV3_PAD)
        self.bn3   = nn.ModuleList([nn.BatchNorm2d(CONV3_OUT, affine=False)
                                     for _ in range(NUM_STEPS)])
        self.lif3  = snn.Leaky(beta=BETA, learn_beta=True, learn_threshold=True)
        self.pool3 = nn.AvgPool2d(POOL_K)

        self.fc_out = nn.Linear(FC_IN, NUM_CLASSES)

        for m in [self.conv1, self.conv2, self.conv3, self.fc_out]:
            fan_in_init_(m, base_std=0.3)

    def forward(self, x):
        B = x.size(0)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_out = torch.zeros(B, NUM_CLASSES, device=x.device)
        out_acc = torch.zeros(B, NUM_CLASSES, device=x.device)

        for t in range(NUM_STEPS):
            x_t = clamp_events(x[:, t])

            cur1 = self.bn1[t](self.conv1(x_t))
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.pool1(spk1)                    # (B, 32, 17, 17)

            cur2 = self.bn2[t](self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.pool2(spk2)                    # (B, 64, 8, 8)

            cur3 = self.bn3[t](self.conv3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3 = self.pool3(spk3)                    # (B, 128, 4, 4)

            flat = spk3.flatten(1)                     # (B, 2048)
            cur_out = self.fc_out(flat)
            mem_out = BETA * mem_out + cur_out
            out_acc += mem_out

        return out_acc / NUM_STEPS


# In[5]:


class SurrogateCSNN(nn.Module):
    """
    Same architecture as CSNN, with fast-sigmoid surrogate gradient for BPTT.
    BatchNorm here uses the standard affine=True (learnable gamma/beta),
    since backprop differentiates through them with no extra cost or
    implementation complexity -- this is the standard BNTT setup used in
    the surrogate-gradient SNN literature.
    """
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.conv1 = nn.Conv2d(C, CONV1_OUT, CONV1_K, padding=CONV1_PAD)
        self.bn1   = nn.ModuleList([nn.BatchNorm2d(CONV1_OUT) for _ in range(NUM_STEPS)])
        self.lif1  = snn.Leaky(beta=BETA, spike_grad=spike_grad,
                                learn_beta=True, learn_threshold=True)
        self.pool1 = nn.AvgPool2d(POOL_K)

        self.conv2 = nn.Conv2d(CONV1_OUT, CONV2_OUT, CONV2_K, padding=CONV2_PAD)
        self.bn2   = nn.ModuleList([nn.BatchNorm2d(CONV2_OUT) for _ in range(NUM_STEPS)])
        self.lif2  = snn.Leaky(beta=BETA, spike_grad=spike_grad,
                                learn_beta=True, learn_threshold=True)
        self.pool2 = nn.AvgPool2d(POOL_K)

        self.conv3 = nn.Conv2d(CONV2_OUT, CONV3_OUT, CONV3_K, padding=CONV3_PAD)
        self.bn3   = nn.ModuleList([nn.BatchNorm2d(CONV3_OUT) for _ in range(NUM_STEPS)])
        self.lif3  = snn.Leaky(beta=BETA, spike_grad=spike_grad,
                                learn_beta=True, learn_threshold=True)
        self.pool3 = nn.AvgPool2d(POOL_K)

        self.fc_out = nn.Linear(FC_IN, NUM_CLASSES)

        for m in [self.conv1, self.conv2, self.conv3, self.fc_out]:
            fan_in_init_(m, base_std=0.3)

    def forward(self, x):
        B = x.size(0)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem_out = torch.zeros(B, NUM_CLASSES, device=x.device)
        out_acc = torch.zeros(B, NUM_CLASSES, device=x.device)

        for t in range(NUM_STEPS):
            x_t = clamp_events(x[:, t])

            cur1 = self.bn1[t](self.conv1(x_t))
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.pool1(spk1)

            cur2 = self.bn2[t](self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.pool2(spk2)

            cur3 = self.bn3[t](self.conv3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3 = self.pool3(spk3)

            flat = spk3.flatten(1)
            cur_out = self.fc_out(flat)
            mem_out = BETA * mem_out + cur_out
            out_acc += mem_out

        return out_acc / NUM_STEPS


criterion = nn.CrossEntropyLoss(reduction="none")

# =============================================================================
# EGGROLL FOR CONV LAYERS — via im2col (unfold)
# =============================================================================
# A conv layer is equivalent to:  W_flat @ unfold(x)  reshaped back to a map.
# W_flat has shape (out_ch, in_ch * k * k). Once we have that matrix view,
# the exact same low-rank correction trick used for linear layers applies:
#     output = W_flat @ patches + scale * A @ (B_lr.T @ patches)
# We compute base conv with F.conv2d (fast, cuDNN), and compute the
# low-rank correction separately via unfold + batched matmul, then add them.
# =============================================================================


# In[6]:


def eggroll_conv2d_cached_input(x_t_shared, patches, base_conv, A, B_lr, rank, sigma, P_sub):
    """
    x_t_shared: (db, Cin, H, W) -- only used for the base conv now
    patches:    (db, Cin*k*k, L) -- precomputed unfold, passed in and reused
    """
    batch = x_t_shared.size(0)
    out_ch = base_conv.out_channels
    scale = sigma / (rank ** 0.5)

    base_out = base_conv(x_t_shared)
    H_out, W_out = base_out.shape[-2:]

    patches_4d = patches.unsqueeze(0).expand(P_sub, -1, -1, -1)
    Bp  = torch.einsum("pir,pbil->pbrl", B_lr, patches_4d)
    ABp = torch.einsum("por,pbrl->pbol", A, Bp)

    corr = (scale * ABp).reshape(P_sub * batch, out_ch, H_out, W_out)
    base_out = base_out.unsqueeze(0).expand(P_sub, -1, -1, -1, -1).reshape(P_sub * batch, out_ch, H_out, W_out)
    return base_out + corr

# In[7]:


def eggroll_conv2d(x, base_conv, A, B_lr, rank, sigma, P_sub, batch):
    """
    x         : (P_sub*batch, in_ch, H, W)
    base_conv : nn.Conv2d module (provides base weight/bias + kernel/padding)
    A         : (P_sub, out_ch, rank)
    B_lr      : (P_sub, in_ch*k*k, rank)
    Returns   : (P_sub*batch, out_ch, H_out, W_out)
    """
    out_ch = base_conv.out_channels
    in_ch = base_conv.in_channels
    k = base_conv.kernel_size[0]
    pad = base_conv.padding[0]
    scale = sigma / (rank ** 0.5)

    # Fast base path
    base_out = base_conv(x)  # (P_sub*batch, out_ch, H_out, W_out)
    H_out, W_out = base_out.shape[-2:]

    # Unfold into patches and reshape into explicit population/batch axes
    patches = F.unfold(x, kernel_size=k, padding=pad)  # (P_sub*batch, in_ch*k*k, L)
    L = patches.shape[-1]

    # Shape safety: the patch count must match the spatial output size.
    expected_L = H_out * W_out
    if L != expected_L:
        raise RuntimeError(
            f"eggroll_conv2d shape mismatch: unfold produced L={L}, "
            f"but conv output has H_out*W_out={expected_L}. "
            f"Check stride/padding/kernel assumptions."
        )

    patches_3d = patches.view(P_sub, batch, in_ch * k * k, L)

    # Low-rank correction:
    #   B_lr projects each patch into rank space, then A maps back to out channels.
    #   Result shape: (P_sub, batch, out_ch, L)
    Bp = torch.einsum("pir,pbil->pbrl", B_lr, patches_3d)
    ABp = torch.einsum("por,pbrl->pbol", A, Bp)

    corr = (ABp * scale).reshape(P_sub * batch, out_ch, H_out, W_out)
    return base_out + corr


# In[8]:


def make_conv_lr_factors(seed, pop_size, rank, in_ch, out_ch, k):
    """Low-rank factors for one conv layer, flattened weight view."""
    g = torch.Generator(device=DEVICE).manual_seed(int(seed))
    A = torch.randn(pop_size, out_ch, rank, generator=g, device=DEVICE)
    B_lr = torch.randn(pop_size, in_ch * k * k, rank, generator=g, device=DEVICE)
    c = torch.randn(pop_size, out_ch, generator=g, device=DEVICE)

    # Shape safety: keep the factor shapes explicit and easy to audit.
    assert A.shape == (pop_size, out_ch, rank)
    assert B_lr.shape == (pop_size, in_ch * k * k, rank)
    assert c.shape == (pop_size, out_ch)
    return A, B_lr, c


# In[9]:


def make_linear_lr_factors(seed, pop_size, rank, in_dim, out_dim):
    g = torch.Generator(device=DEVICE).manual_seed(int(seed))
    A = torch.randn(pop_size, out_dim, rank, generator=g, device=DEVICE)
    B_lr = torch.randn(pop_size, in_dim, rank, generator=g, device=DEVICE)
    c = torch.randn(pop_size, out_dim, generator=g, device=DEVICE)

    # Shape safety.
    assert A.shape == (pop_size, out_dim, rank)
    assert B_lr.shape == (pop_size, in_dim, rank)
    assert c.shape == (pop_size, out_dim)
    return A, B_lr, c


# In[10]:


def make_all_factors(seed, pop_size, rank):
    s1, s2, s3, s4 = seed, seed + 7919, seed + 15991, seed + 23993
    A1, B1, c1 = make_conv_lr_factors(s1, pop_size, rank, C,         CONV1_OUT, CONV1_K)
    A2, B2, c2 = make_conv_lr_factors(s2, pop_size, rank, CONV1_OUT, CONV2_OUT, CONV2_K)
    A3, B3, c3 = make_conv_lr_factors(s3, pop_size, rank, CONV2_OUT, CONV3_OUT, CONV3_K)
    A4, B4, c4 = make_linear_lr_factors(s4, pop_size, rank, FC_IN, NUM_CLASSES)
    return (A1,B1,c1), (A2,B2,c2), (A3,B3,c3), (A4,B4,c4)


# In[11]:


def eggroll_forward_subbatch(data, base_net, factors, P_sub, rank, sigma, patches_per_t):
    (A1,B1,c1),(A2,B2,c2),(A3,B3,c3),(A4,B4,c4) = factors
    batch = data.size(0)
    db = batch // P_sub

    sigma1 = sigma * INIT_STD_CONV1
    sigma2 = sigma * INIT_STD_CONV2
    sigma3 = sigma * INIT_STD_CONV3
    sigma4 = sigma * INIT_STD_FC
    scale4 = sigma4 / (rank ** 0.5)

    # Keep these in fp32 explicitly -- they integrate over 20 timesteps,
    # and small per-step rounding errors in bf16 would compound.
    mem1    = torch.zeros(batch, CONV1_OUT, H,  W,  device=DEVICE, dtype=torch.float32)
    mem2    = torch.zeros(batch, CONV2_OUT, H2, W2, device=DEVICE, dtype=torch.float32)
    mem3    = torch.zeros(batch, CONV3_OUT, H3, W3, device=DEVICE, dtype=torch.float32)
    mem_out = torch.zeros(batch, NUM_CLASSES, device=DEVICE, dtype=torch.float32)
    out_acc = torch.zeros(batch, NUM_CLASSES, device=DEVICE, dtype=torch.float32)

    bc1 = (sigma1 / (CONV1_OUT ** 0.5) * c1).view(P_sub, 1, CONV1_OUT, 1, 1)
    bc2 = (sigma2 / (CONV2_OUT ** 0.5) * c2).view(P_sub, 1, CONV2_OUT, 1, 1)
    bc3 = (sigma3 / (CONV3_OUT ** 0.5) * c3).view(P_sub, 1, CONV3_OUT, 1, 1)
    bc4 = (sigma4 / (NUM_CLASSES ** 0.5) * c4).view(P_sub, 1, NUM_CLASSES)

    lif1, lif2, lif3 = base_net.lif1, base_net.lif2, base_net.lif3

    for t in range(NUM_STEPS):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x_t = clamp_events(data[:, t])
            x_t_shared = x_t[:db]

            raw_cur1 = eggroll_conv2d_cached_input(x_t_shared, patches_per_t[t], base_net.conv1, A1, B1, rank, sigma1, P_sub)
            cur1 = (raw_cur1.view(P_sub, db, CONV1_OUT, H, W) + bc1).reshape(P_sub * db, CONV1_OUT, H, W)
            cur1 = base_net.bn1[t](cur1)

            spk1, mem1 = lif1(cur1.float(), mem1)   # single fp32 cast, LIF still inside the block
            spk1 = base_net.pool1(spk1)

            raw_cur2 = eggroll_conv2d(spk1, base_net.conv2, A2, B2, rank, sigma2, P_sub, db)
            cur2 = (raw_cur2.view(P_sub, db, CONV2_OUT, H2, W2) + bc2).reshape(P_sub * db, CONV2_OUT, H2, W2)
            cur2 = base_net.bn2[t](cur2)
            spk2, mem2 = lif2(cur2.float(), mem2)
            spk2 = base_net.pool2(spk2)

            raw_cur3 = eggroll_conv2d(spk2, base_net.conv3, A3, B3, rank, sigma3, P_sub, db)
            cur3 = (raw_cur3.view(P_sub, db, CONV3_OUT, H3, W3) + bc3).reshape(P_sub * db, CONV3_OUT, H3, W3)
            cur3 = base_net.bn3[t](cur3)
            spk3, mem3 = lif3(cur3.float(), mem3)
            spk3 = base_net.pool3(spk3)

            flat = spk3.flatten(1)
            flat_3d = flat.view(P_sub, db, FC_IN)
            base_cur4 = flat @ base_net.fc_out.weight.T + base_net.fc_out.bias
            Bp4  = torch.einsum("pir,pbi->pbr", B4, flat_3d)
            ABp4 = torch.einsum("por,pbr->pbo", A4, Bp4)
            cur4 = base_cur4 + (scale4 * ABp4 + bc4).reshape(P_sub * db, NUM_CLASSES)

        mem_out = BETA * mem_out + cur4.float()
        out_acc += mem_out

    return out_acc / NUM_STEPS   # dedented -- must run after all NUM_STEPS timesteps, not just t=0


# In[12]:


def eggroll_grad_conv(f_diff, A, B_lr, rank, sigma, pop_size):
    """Gradient for a conv layer's flattened weight, using (diag(f)A)^T B."""
    f  = torch.from_numpy(f_diff).to(DEVICE).float()
    sc = 1.0 / (rank ** 0.5)
    nm = 1.0 / (2.0 * pop_size * sigma)
    fA = f.view(pop_size, 1, 1) * A
    grad_flat = torch.einsum("por,pir->oi", fA, B_lr) * sc * nm  # (out_ch, in_ch*k*k)
    return grad_flat

# In[13]:


def eggroll_grad_linear(f_diff, A, B_lr, rank, sigma, pop_size):
    f  = torch.from_numpy(f_diff).to(DEVICE).float()
    sc = 1.0 / (rank ** 0.5)
    nm = 1.0 / (2.0 * pop_size * sigma)
    fA = f.view(pop_size, 1, 1) * A
    return torch.einsum("por,pir->oi", fA, B_lr) * sc * nm

# In[14]:


def eggroll_grad_bias(f_diff, c, dim, sigma, pop_size):
    """
    Forward pass injects bias noise with std = sigma / sqrt(dim) (see bc1/bc2/bc3
    in eggroll_forward_subbatch). For an unbiased ES gradient estimate, the
    denominator must match the ACTUAL injected noise std, not sigma alone:

        grad = (1 / (2N * sigma_eff)) * sum_i f_i * c_i
             = (1 / (2N * sigma/sqrt(dim))) * sum_i f_i * c_i
             = (sqrt(dim) / (2N*sigma)) * sum_i f_i * c_i

    The old version divided by sqrt(dim) here too — same direction as the
    forward pass — which is backwards: it needs to be multiplied here to
    invert the forward-pass shrinkage. The old code understated bias
    gradients by a factor of `dim` (32, 64, or 10 depending on layer).
    Adam's per-coordinate normalisation papered over this, which is why
    training still worked despite the bug.
    """
    f  = torch.from_numpy(f_diff).to(DEVICE).float()
    nm = 1.0 / (2.0 * pop_size * sigma)
    return (f.view(pop_size, 1) * c).sum(0) * nm * (dim ** 0.5)

# =============================================================================
# CENTERED RANKS (fixes the z-score bug found earlier)
# =============================================================================


# In[15]:


def centered_ranks(x):
    x     = np.asarray(x, dtype=np.float64)
    ranks = np.empty_like(x)
    ranks[np.argsort(x)] = np.arange(len(x), dtype=np.float64)
    if len(x) > 1:
        ranks /= (len(x) - 1)
    ranks -= 0.5
    return ranks



# In[16]:


# =============================================================================
# UTILITIES
# =============================================================================

@torch.no_grad()
def get_eval_batches(loader, k):
    batches, it = [], iter(loader)
    for _ in range(k):
        try:   data, targets = next(it)
        except StopIteration:
            it = iter(loader); data, targets = next(it)
        batches.append((data.to(DEVICE), targets.to(DEVICE)))
    return batches



# In[17]:


@torch.no_grad()
def accuracy_on_loader(net, loader):
    net.eval()
    total = correct = 0
    for data, targets in loader:
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        logits  = net(data)
        _, pred = logits.max(1)
        total   += targets.size(0)
        correct += (pred == targets).sum().item()
    return 100.0 * correct / total



# In[18]:


@torch.no_grad()
def evaluate_eggroll(base_net, batches, rank, sigma, population, seed_pos):
    base_net.train()
    factors_full = make_all_factors_depth(seed_pos, population, rank, base_net, device=DEVICE)
    rewards = {}

    # Precompute layer-1 unfold once per generation -- it's identical across
    # every population sub-batch chunk AND across the pos/neg sign loop,
    # since it only depends on the raw input data and conv1's kernel/padding,
    # neither of which changes across those loops.
    cached_patches = []
    for data, targets in batches:
        db = data.size(0)
        patches_per_t = [
            F.unfold(clamp_events(data[:, t]), kernel_size=CONV1_K, padding=CONV1_PAD)
            for t in range(NUM_STEPS)
        ]
        cached_patches.append(patches_per_t)

    for sign_str, sign in [("pos", 1.0), ("neg", -1.0)]:
        #torch.cuda.empty_cache()
        sf = tuple((sign * A, B, sign * c) for (A, B, c) in factors_full)
        total = torch.zeros(population, device=DEVICE)
        for p0 in range(0, population, POP_SUBBATCH):
            p1  = min(p0 + POP_SUBBATCH, population)
            P_s = p1 - p0
            fsub = tuple(tuple(t[p0:p1] for t in layer) for layer in sf)
            sub = torch.zeros(P_s, device=DEVICE)
            for data, targets in batches:
                db     = data.size(0)
                x_tile = (data.unsqueeze(0)
                              .expand(P_s, -1, -1, -1, -1, -1)
                              .reshape(P_s * db, NUM_STEPS, C, H, W))
                t_tile = targets.unsqueeze(0).expand(P_s, -1).reshape(P_s * db)
                logits = depth_forward_subbatch(x_tile, base_net, fsub, P_s, rank, sigma, NUM_STEPS, BETA, device=DEVICE)
                losses = criterion(logits, t_tile).view(P_s, db).mean(dim=1)
                sub   -= losses
            total[p0:p1] = sub / len(batches)
        rewards[sign_str] = total.cpu().numpy()
    return rewards["pos"], rewards["neg"]

# In[19]:


def run_eggroll_csnn(generations, population, rank, sigma0, sigma_decay, sigma_min,
                     lr, eval_batches_k, seed, net=None):
    torch.manual_seed(seed); np.random.seed(seed)
    if net is None:
        net = CSNN().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.RandomState(seed + 1000)

    fitness_hist, val_hist, val_gens, gen_times, sigma_hist = [], [], [], [], []
    sigma_t = sigma0
    print(f"\n[EGGROLL CSNN] rank={rank} pop={population} seed={seed}")

    for gen in range(generations):
        t0 = time.perf_counter()
        eval_data = get_eval_batches(train_loader, eval_batches_k)
        seed_pos  = int(rng.randint(0, 2**31))

        r_pos, r_neg = evaluate_eggroll(net, eval_data, rank, sigma_t, population, seed_pos)
        ranks_all = centered_ranks(np.concatenate([r_pos, r_neg]))
        rp, rn    = ranks_all[:population], ranks_all[population:]
        f_diff    = rp - rn

        factors = make_all_factors_depth(seed_pos, population, rank, net, device=DEVICE)

        opt.zero_grad()
        for i in range(net.depth):
            A, B, c = factors[i]
            gW = eggroll_grad_conv(f_diff, A, B, rank, sigma_t, population)
            gb = eggroll_grad_bias(f_diff, c, net.chans[i+1], sigma_t, population)
            net.convs[i].weight.grad = -gW.view_as(net.convs[i].weight)
            net.convs[i].bias.grad   = -gb
        A4, B4, c4 = factors[net.depth]
        gW4 = eggroll_grad_linear(f_diff, A4, B4, rank, sigma_t, population)
        gb4 = eggroll_grad_bias(f_diff, c4, net.num_classes, sigma_t, population)
        net.fc_out.weight.grad = -gW4.view_as(net.fc_out.weight)
        net.fc_out.bias.grad   = -gb4
        opt.step()

        gen_times.append(time.perf_counter() - t0)
        fitness_hist.append(float(np.mean(r_pos)))
        sigma_hist.append(sigma_t)

        # Validation is a full pass over the val set — expensive. Only run it
        # periodically (the old code ran it every generation).
        if gen % VAL_EVERY == 0 or gen == generations - 1:
            v = accuracy_on_loader(net, val_loader)
            net.train()                       # accuracy_on_loader puts net in eval()
            val_hist.append(v)
            val_gens.append(gen)



        sigma_t = max(sigma_t*sigma_decay, SIGMA_MIN)

        if gen % 20 == 0 or gen == generations - 1:
            last_val = val_hist[-1] if val_hist else float("nan")
            print(f"  gen {gen:4d} | fit {fitness_hist[-1]:.3f} "
                  f"| val {last_val:.2f}% | sigma {sigma_t:.5f} | {gen_times[-1]:.2f}s")

    return dict(fitness=fitness_hist, val_acc=val_hist, val_gens=val_gens,
                gen_times=gen_times, sigma_hist=sigma_hist,
                best_val=max(val_hist) if val_hist else 0.0,
                stopped_early=False, stop_gen=generations - 1,
                test_acc=accuracy_on_loader(net, test_loader), net=net)


# In[20]:


# =============================================================================
# SURROGATE-GRADIENT BPTT BASELINE
# =============================================================================
def run_surrogate_csnn(epochs, lr, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = SurrogateCSNN().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    val_hist, epoch_times = [], []
    print(f"\n[Surrogate BPTT CSNN] seed={seed}")

    for ep in range(epochs):
        t0 = time.perf_counter(); net.train()
        for data, targets in train_loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            logits = net(data)
            loss   = loss_fn(logits, targets)
            opt.zero_grad(); loss.backward(); opt.step()
        epoch_times.append(time.perf_counter() - t0)
        val_hist.append(accuracy_on_loader(net, val_loader))
        print(f"  epoch {ep:02d} | val {val_hist[-1]:.2f}% | {epoch_times[-1]:.2f}s")

    return dict(val_acc=val_hist, epoch_times=epoch_times,
                test_acc=accuracy_on_loader(net, test_loader), net=net)



# In[ ]:


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("Training EGGROLL CSNN on N-MNIST")
    print("=" * 60)
    eggroll_result = run_eggroll_csnn(
        generations=GENERATIONS, population=POPULATION, rank=RANK,
        sigma0=SIGMA0, sigma_decay=SIGMA_DECAY, sigma_min=SIGMA_MIN,
        lr=LR_ES, eval_batches_k=EVAL_BATCHES_K, seed=SEED)

    print("\n" + "=" * 60)
    print("Training Surrogate-Gradient CSNN baseline on N-MNIST")
    print("=" * 60)
    bptt_result = run_surrogate_csnn(epochs=BPTT_EPOCHS, lr=LR_BPTT, seed=SEED)

    # ── Comparison ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL COMPARISON — N-MNIST CSNN")
    print("=" * 60)
    print(f"EGGROLL  (rank={RANK}, {GENERATIONS} gens) : "
          f"{eggroll_result['test_acc']:.2f}%  "
          f"| {np.mean(eggroll_result['gen_times']):.2f}s/gen")
    print(f"Surrogate BPTT ({BPTT_EPOCHS} epochs)        : "
          f"{bptt_result['test_acc']:.2f}%  "
          f"| {np.mean(bptt_result['epoch_times']):.2f}s/epoch")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # val_acc is now sampled every VAL_EVERY generations, so plot against the
    # actual generation indices stored in val_gens (not 0..len-1).
    axes[0].plot(eggroll_result["val_gens"], eggroll_result["val_acc"],
                 color="#d73027", lw=1.5, marker="o", ms=3, label="EGGROLL")
    bptt_x = np.linspace(0, GENERATIONS, len(bptt_result["val_acc"]))
    axes[0].plot(bptt_x, bptt_result["val_acc"], color="black", ls="--", lw=1.5, label="Surrogate BPTT")
    axes[0].set(title="Validation accuracy — N-MNIST CSNN",
                xlabel="EGGROLL generation (BPTT epoch rescaled)", ylabel="Accuracy (%)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(eggroll_result["sigma_hist"], color="#2c7bb6", lw=1.5)
    axes[1].axhline(SIGMA_MIN, ls="--", color="gray", label=f"sigma_min={SIGMA_MIN}")
    axes[1].set(title="Sigma decay schedule (relative)", xlabel="Generation", ylabel="Sigma")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle("EGGROLL CSNN vs Surrogate BPTT — N-MNIST", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "csnn_comparison.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # ── Save ──────────────────────────────────────────────────────────────
    def cast(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, dict): return {str(k): cast(v) for k, v in o.items() if k != "net"}
        if isinstance(o, list): return [cast(v) for v in o]
        return o

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump({"eggroll": cast(eggroll_result), "bptt": cast(bptt_result)}, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()


# In[ ]:



