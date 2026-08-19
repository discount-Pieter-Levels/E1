"""
Depth-parameterized CSNN + EGGROLL, for E1 (depth-scaling study).

Drop-in generalization of the hardcoded 3-block CSNN to an arbitrary depth D.
Everything (channels, spatial chain, factors, forward-subbatch, gradient
reconstruction) is derived from a per-depth channel list, so depth is the ONLY
thing that changes between conditions -- the requirement for a clean scaling curve.

This module is self-contained and tested standalone. Integrate by replacing the
corresponding notebook cells, OR import these functions.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import snntorch as snn


# ============================================================================
# DEPTH-PARAMETERIZED ARCHITECTURE SPEC
# ============================================================================
def make_channel_list(depth, base_width=32, max_width=256, in_ch=2):
    """
    Channel progression for a depth-D net. Doubles width each block up to
    max_width, then holds. Width POLICY is fixed across depths so that depth is
    isolated from width (a shallow and a deep net use the same rule, just more/
    fewer blocks). Returns [in_ch, w1, w2, ..., wD].
    """
    chans = [in_ch]
    w = base_width
    for _ in range(depth):
        chans.append(min(w, max_width))
        w *= 2
    return chans


def make_downsample_schedule(depth, in_hw, target_hw=4):
    """
    Decide which blocks downsample (stride/pool by 2) so that after D blocks the
    feature map is ~target_hw, REGARDLESS of depth or input size. This keeps
    FC_IN in a controlled band across depths, so depth is not confounded with
    readout size (the confound the strided-conv experiment showed matters).

    Returns a list of D booleans: does block i downsample?
    Strategy: we need ceil(log2(in_hw/target_hw)) downsampling steps; place them
    in the LAST blocks (downsample late, so early blocks process at full res).
    If more blocks than needed steps -> extra blocks are stride-1. If fewer
    blocks than needed steps -> downsample every block (can't reach target, but
    monotonic).
    """
    import math
    n_ds_needed = max(0, int(math.ceil(math.log2(max(1, in_hw / target_hw)))))
    n_ds = min(n_ds_needed, depth)
    # place the n_ds downsampling ops in the last n_ds blocks
    sched = [False] * depth
    for i in range(depth - n_ds, depth):
        sched[i] = True
    return sched


def compute_spatial_chain(in_hw, kernel, pad, ds_sched):
    """Return (conv_out_hw, post_pool_hw) per block.
    Convs are stride-1 same-pad, so conv OUTPUT size == that block's INPUT size.
    Pooling (if any) happens AFTER the LIF, halving the size for the NEXT block.
      conv_out[i]  = input size to block i (in_hw for i=0, else post_pool[i-1])
      post_pool[i] = conv_out[i] // 2 if block i downsamples, else conv_out[i]
    raw/cur/membrane tensors live at conv_out[i]; the next block sees post_pool[i]."""
    conv_out, post_pool = [], []
    hw = in_hw
    for ds in ds_sched:
        conv_out.append(hw)                    # conv output = current input size
        hw = hw // 2 if ds else hw             # pool after LIF -> next block input
        post_pool.append(hw)
    return conv_out, post_pool


class DepthCSNN(nn.Module):
    """
    Depth-D conv-LIF network. Same block design as the original 3-block CSNN
    (Conv -> BN(affine=False) -> LIF -> optional AvgPool downsample), just
    parameterized over depth. Plain forward() used for validation/test only;
    ES trains via depth_forward_subbatch.
    """
    def __init__(self, depth, in_ch, in_hw, num_classes, num_steps, beta,
                 kernel=3, pad=1, base_width=32, max_width=256, base_std=0.3):
        super().__init__()
        self.depth = depth
        self.num_steps = num_steps
        self.beta = beta
        self.kernel = kernel
        self.pad = pad
        self.chans = make_channel_list(depth, base_width, max_width, in_ch)
        self.ds_sched = make_downsample_schedule(depth, in_hw)
        self.conv_out_hw, self.post_pool_hw = compute_spatial_chain(in_hw, kernel, pad, self.ds_sched)
        self.spatial = self.post_pool_hw       # back-compat alias (post-pool sizes)
        self.fc_in = self.chans[-1] * self.post_pool_hw[-1] * self.post_pool_hw[-1]
        self.num_classes = num_classes

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()   # ModuleList of ModuleList(per-timestep BN)
        self.lifs  = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(depth):
            cin, cout = self.chans[i], self.chans[i + 1]
            self.convs.append(nn.Conv2d(cin, cout, kernel, stride=1, padding=pad))
            self.bns.append(nn.ModuleList(
                [nn.BatchNorm2d(cout, affine=False) for _ in range(num_steps)]))
            self.lifs.append(snn.Leaky(beta=beta, learn_beta=True, learn_threshold=True))
            self.pools.append(nn.AvgPool2d(2) if self.ds_sched[i] else nn.Identity())
        self.fc_out = nn.Linear(self.fc_in, num_classes)

        # fan-in init on all convs + readout
        for m in list(self.convs) + [self.fc_out]:
            if isinstance(m, nn.Conv2d):
                fi = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            else:
                fi = m.in_features
            nn.init.normal_(m.weight, 0.0, base_std / (fi ** 0.5))

    def forward(self, x):
        B = x.size(0)
        mems = [lif.init_leaky() for lif in self.lifs]
        mem_out = torch.zeros(B, self.num_classes, device=x.device)
        out_acc = torch.zeros(B, self.num_classes, device=x.device)
        for t in range(self.num_steps):
            z = x[:, t].clamp(0, 1)
            for i in range(self.depth):
                cur = self.bns[i][t](self.convs[i](z))
                spk, mems[i] = self.lifs[i](cur, mems[i])
                z = self.pools[i](spk)
            flat = z.flatten(1)
            mem_out = self.beta * mem_out + self.fc_out(flat)
            out_acc += mem_out
        return out_acc / self.num_steps


# quick shape test across depths
if __name__ == "__main__":
    for D in [2, 3, 4, 5, 6]:
        for in_hw, in_ch, ncls in [(34, 2, 10), (128, 2, 11)]:
            net = DepthCSNN(D, in_ch, in_hw, ncls, num_steps=3, beta=0.95)
            x = torch.rand(2, 3, in_ch, in_hw, in_hw)
            out = net(x)
            assert out.shape == (2, ncls), out.shape
            n_params = sum(p.numel() for p in net.parameters())
            print(f"D={D} in={in_hw}x{in_hw} chans={net.chans} ds={net.ds_sched} "
                  f"spatial={net.spatial} fc_in={net.fc_in} params={n_params}")
    print("DEPTH ARCHITECTURE SHAPE TEST PASS")


# ============================================================================
# DEPTH-GENERIC EGGROLL MACHINERY
# (reuses the per-layer helpers eggroll_conv2d / _cached_input / grad fns,
#  which are already depth-agnostic -- only the block LOOP needed generalizing)
# ============================================================================

def make_conv_lr_factors(seed, pop, rank, in_ch, out_ch, k, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(pop, out_ch, rank, generator=g)
    B = torch.randn(pop, in_ch * k * k, rank, generator=g)
    c = torch.randn(pop, out_ch, generator=g)
    return A.to(device), B.to(device), c.to(device)

def make_linear_lr_factors(seed, pop, rank, in_f, out_f, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(pop, out_f, rank, generator=g)
    B = torch.randn(pop, in_f, rank, generator=g)
    c = torch.randn(pop, out_f, generator=g)
    return A.to(device), B.to(device), c.to(device)

def make_all_factors_depth(seed, pop, rank, net, device="cpu"):
    """One (A,B,c) per conv block + one for the linear readout. Deterministic
    per-block seeds so factors regenerate identically for grad reconstruction."""
    factors = []
    for i in range(net.depth):
        cin, cout = net.chans[i], net.chans[i + 1]
        s = seed + 7919 * (i + 1)
        factors.append(make_conv_lr_factors(s, pop, rank, cin, cout, net.kernel, device))
    s = seed + 7919 * (net.depth + 1)
    factors.append(make_linear_lr_factors(s, pop, rank, net.fc_in, net.num_classes, device))
    return factors  # length depth+1; last is the linear readout

def eggroll_conv2d_cached_input(x_t_shared, patches, base_conv, A, B_lr, rank, sigma, P_sub):
    batch = x_t_shared.size(0); out_ch = base_conv.out_channels
    scale = sigma / (rank ** 0.5)
    base_out = base_conv(x_t_shared); H_out, W_out = base_out.shape[-2:]
    patches_4d = patches.unsqueeze(0).expand(P_sub, -1, -1, -1)
    Bp  = torch.einsum("pir,pbil->pbrl", B_lr, patches_4d)
    ABp = torch.einsum("por,pbrl->pbol", A, Bp)
    corr = (scale * ABp).reshape(P_sub * batch, out_ch, H_out, W_out)
    base_out = base_out.unsqueeze(0).expand(P_sub, -1, -1, -1, -1).reshape(P_sub * batch, out_ch, H_out, W_out)
    return base_out + corr

def eggroll_conv2d(x, base_conv, A, B_lr, rank, sigma, P_sub, batch):
    out_ch = base_conv.out_channels; in_ch = base_conv.in_channels
    k = base_conv.kernel_size[0]; pad = base_conv.padding[0]; stride = base_conv.stride[0]
    scale = sigma / (rank ** 0.5)
    base_out = base_conv(x); H_out, W_out = base_out.shape[-2:]
    patches = F.unfold(x, kernel_size=k, padding=pad, stride=stride)
    L = patches.shape[-1]; assert L == H_out * W_out
    patches_3d = patches.view(P_sub, batch, in_ch * k * k, L)
    Bp  = torch.einsum("pir,pbil->pbrl", B_lr, patches_3d)
    ABp = torch.einsum("por,pbrl->pbol", A, Bp)
    corr = (scale * ABp).reshape(P_sub * batch, out_ch, H_out, W_out)
    return base_out + corr

def depth_forward_subbatch(data, net, factors, P_sub, rank, sigma, num_steps,
                           beta, base_std=0.3, device="cpu"):
    """Depth-generic EGGROLL forward over the population sub-batch. Mirrors the
    3-block version but loops over net.depth conv blocks."""
    batch = data.size(0); db = batch // P_sub; D = net.depth
    # per-layer sigma from fan-in (same rule as fan_in_init)
    def layer_sigma(cin, k): return sigma * (base_std / ((cin * k * k) ** 0.5))
    sigmas = [layer_sigma(net.chans[i], net.kernel) for i in range(D)]
    sigma_fc = sigma * (base_std / (net.fc_in ** 0.5))
    scale_fc = sigma_fc / (rank ** 0.5)

    # membrane potentials at each block's conv-output spatial size
    mems = [torch.zeros(batch, net.chans[i + 1], net.conv_out_hw[i], net.conv_out_hw[i],
                        device=device, dtype=torch.float32) for i in range(D)]
    mem_out = torch.zeros(batch, net.num_classes, device=device, dtype=torch.float32)
    out_acc = torch.zeros(batch, net.num_classes, device=device, dtype=torch.float32)

    # bias-correction terms per block
    bcs = []
    for i in range(D):
        cout = net.chans[i + 1]
        A, B, c = factors[i]
        bcs.append((sigmas[i] / (cout ** 0.5) * c).view(P_sub, 1, cout, 1, 1))
    A4, B4, c4 = factors[D]
    bc4 = (sigma_fc / (net.num_classes ** 0.5) * c4).view(P_sub, 1, net.num_classes)

    for t in range(num_steps):
        x_t = data[:, t].clamp(0, 1)         # (P_sub*db, Cin, H, W)
        for i in range(D):
            A, B, c = factors[i]
            cout, sh = net.chans[i + 1], net.conv_out_hw[i]
            if i == 0:
                # Block 0: every population member sees the SAME raw frame, so the
                # base conv + unfold run once on the db-sized shared input and are
                # broadcast across P_sub (the cached-input trick).
                x_shared = x_t[:db]
                patches = F.unfold(x_shared, kernel_size=net.kernel,
                                   padding=net.pad, stride=1)
                raw = eggroll_conv2d_cached_input(x_shared, patches, net.convs[i],
                                                  A, B, rank, sigmas[i], P_sub)
            else:
                # Later blocks: each member has its own activations (P_sub*db rows).
                raw = eggroll_conv2d(z, net.convs[i], A, B, rank, sigmas[i], P_sub, db)
            cur = (raw.view(P_sub, db, cout, sh, sh) + bcs[i]).reshape(P_sub * db, cout, sh, sh)
            cur = net.bns[i][t](cur)
            spk, mems[i] = net.lifs[i](cur.float(), mems[i])
            z = net.pools[i](spk)
        flat = z.flatten(1)                  # (P_sub*db, fc_in)
        flat_3d = flat.view(P_sub, db, net.fc_in)
        base4 = flat @ net.fc_out.weight.T + net.fc_out.bias
        Bp4  = torch.einsum("pir,pbi->pbr", B4, flat_3d)
        ABp4 = torch.einsum("por,pbr->pbo", A4, Bp4)
        cur4 = base4 + (scale_fc * ABp4 + bc4).reshape(P_sub * db, net.num_classes)
        mem_out = beta * mem_out + cur4.float()
        out_acc += mem_out
    return out_acc / num_steps


def eggroll_grad_conv(f_diff, A, B_lr, rank, sigma, pop_size, device="cpu"):
    f  = torch.from_numpy(f_diff).to(device).float()
    sc = 1.0 / (rank ** 0.5); nm = 1.0 / (2.0 * pop_size * sigma)
    fA = f.view(pop_size, 1, 1) * A
    return torch.einsum("por,pir->oi", fA, B_lr) * sc * nm

# ---- test the full depth-generic ES path across depths ----
if __name__ == "__main__":
    print("\n--- depth-generic EGGROLL forward test ---")
    for D in [2, 3, 4, 5]:
        net = DepthCSNN(D, in_ch=2, in_hw=34, num_classes=10, num_steps=3, beta=0.95).train()
        pop, rank, P_sub, db = 8, 8, 4, 2
        factors_full = make_all_factors_depth(123, pop, rank, net)
        assert len(factors_full) == D + 1, f"expected {D+1} factor tuples, got {len(factors_full)}"
        # In the real pipeline evaluate_eggroll slices the population into sub-batches
        # of P_sub members; the forward receives P_sub-sized factors. Slice here.
        factors = [(A[:P_sub], B[:P_sub], c[:P_sub]) for (A,B,c) in factors_full]
        data = torch.rand(P_sub * db, 3, 2, 34, 34)
        out = depth_forward_subbatch(data, net, factors, P_sub, rank, 0.05,
                                     num_steps=3, beta=0.95)
        assert out.shape == (P_sub * db, 10), out.shape
        # grad reconstruction shape check for each conv block
        f_diff = (np.arange(pop) - pop / 2).astype("float64")
        for i in range(D):
            A, B, c = factors_full[i]
            g = eggroll_grad_conv(f_diff, A, B, rank, 0.05, pop)
            expected = net.convs[i].weight.numel()
            assert g.numel() == expected, f"D={D} block{i} grad {g.numel()} != {expected}"
        print(f"D={D}: forward {tuple(out.shape)}, {D+1} factor sets, all grad shapes OK")
    print("DEPTH-GENERIC EGGROLL PATH TEST PASS")


