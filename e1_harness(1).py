"""
E1 — Depth Scaling Experiment harness.

Sweeps network depth (holding population, rank, LR, and all else FIXED), runs
EGGROLL training at each depth for multiple seeds, and records everything the
depth-scaling analysis needs. Designed to drop alongside depth_eggroll.py and the
existing training loop / data loaders.

WHAT E1 MEASURES (per depth, per seed):
  - final test accuracy + best validation accuracy
  - full validation trajectory (curve shape is the central finding)
  - generations-to-threshold (to 50/70/85/90%) — convergence speed vs depth
  - DID IT TRAIN AT ALL: does val move off chance? (the "breakdown depth" signal)
  - peak GPU memory + wall-clock per generation (efficiency vs depth)
  - parameter count, fc_in, channel list, downsample schedule (architecture record)
  - per-block early-vs-late weight movement (the ES gradient-vanishing analogue;
    supports the E3 mechanism analysis later)

CRITICAL CONTROL (see paper notes): POPULATION IS HELD FIXED across all depths and
must be chosen high enough not to starve the DEEPEST net. Do NOT vary it here — it
becomes its own axis in E3. Set E1_POPULATION once below.
"""
import os, json, time, numpy as np, torch


def generations_to_threshold(val_hist, val_gens, thresholds=(50, 70, 85, 90)):
    """First generation at which val accuracy first reaches each threshold.
    None if never reached (a key signal: deep nets may never cross 70%)."""
    out = {}
    for th in thresholds:
        hit = next((g for g, v in zip(val_gens, val_hist) if v >= th), None)
        out[f"gen_to_{th}"] = hit
    return out


def trained_at_all(val_hist, chance, margin=5.0):
    """Did validation move meaningfully off chance? The breakdown-depth signal:
    a net that never beats chance+margin failed to train under ES at this depth."""
    if not val_hist:
        return False
    return bool(max(val_hist) > (chance + margin))


def snapshot_block_norms(net):
    """Per-conv-block weight L2 norm — compared before/after training to measure
    how much each block moved. Early blocks that don't move in deep nets are the
    ES analogue of gradient vanishing (feeds the E3 mechanism figure)."""
    return [float(net.convs[i].weight.norm().item()) for i in range(net.depth)]


def run_depth_sweep(
        depths,                    # e.g. [2,3,4,5,6]
        seeds,                     # e.g. [0,1,2]
        build_net,                 # fn(depth, seed) -> DepthCSNN on DEVICE (fresh init)
        train_fn,                  # fn(net, generations, population, rank, lr, seed) -> result dict
        *,
        generations, population, rank, lr,
        chance,                    # 100/num_classes, for the trained-at-all test
        out_dir, dataset_tag,
        thresholds=(50, 70, 85, 90)):
    """
    Runs the full depth × seed grid. train_fn is your EGGROLL training loop adapted
    to accept a prebuilt net and return at least:
        {val_acc, val_gens, test_acc, best_val, gen_times, peak_mem_mb, n_params,
         stopped_early, stop_gen, net}
    Writes one JSON per (depth, seed) and an aggregate per depth.
    """
    os.makedirs(out_dir, exist_ok=True)
    all_records = []

    for D in depths:
        per_seed = []
        for sd in seeds:
            net = build_net(D, sd)
            arch = dict(depth=D, chans=net.chans, ds_sched=net.ds_sched,
                        conv_out_hw=net.conv_out_hw, post_pool_hw=net.post_pool_hw,
                        fc_in=net.fc_in,
                        n_params=sum(p.numel() for p in net.parameters()))
            norms_before = snapshot_block_norms(net)

            print(f"\n=== [{dataset_tag}] depth D={D} seed={sd} "
                  f"pop={population} rank={rank} params={arch['n_params']} ===")
            t0 = time.time()
            res = train_fn(net, generations=generations, population=population,
                           rank=rank, lr=lr, seed=sd)
            wall = time.time() - t0

            norms_after = snapshot_block_norms(res.get("net", net))
            block_movement = [a - b for a, b in zip(norms_after, norms_before)]

            g2t = generations_to_threshold(res["val_acc"], res["val_gens"], thresholds)
            rec = {
                "dataset": dataset_tag, "depth": D, "seed": sd,
                "population": population, "rank": rank, "lr": lr,
                "generations": generations,
                "architecture": arch,
                "test_acc": float(res["test_acc"]),
                "best_val": float(res["best_val"]),
                "val_acc": [float(v) for v in res["val_acc"]],
                "val_gens": [int(g) for g in res["val_gens"]],
                **g2t,
                "trained_at_all": trained_at_all(res["val_acc"], chance),
                "chance": chance,
                "peak_mem_mb": float(res.get("peak_mem_mb", float("nan"))),
                "s_per_gen": float(np.mean(res["gen_times"])) if res.get("gen_times") else float("nan"),
                "wall_clock_s": wall,
                "stopped_early": bool(res.get("stopped_early", False)),
                "stop_gen": int(res.get("stop_gen", generations - 1)),
                # E3 mechanism support: per-block weight movement
                "block_norms_before": norms_before,
                "block_norms_after": norms_after,
                "block_movement": block_movement,
            }
            path = os.path.join(out_dir, f"e1_{dataset_tag}_D{D}_seed{sd}.json")
            with open(path, "w") as f:
                json.dump(rec, f, indent=2)
            print(f"[E1] D={D} seed={sd} | test {rec['test_acc']:.2f}% "
                  f"best_val {rec['best_val']:.2f}% trained={rec['trained_at_all']} "
                  f"gen_to_85={rec['gen_to_85']} peakmem {rec['peak_mem_mb']:.0f}MB "
                  f"{rec['s_per_gen']:.1f}s/gen")
            per_seed.append(rec)
            all_records.append(rec)

        # aggregate this depth across seeds
        def ms(key):
            vals = [r[key] for r in per_seed if r[key] is not None]
            if not vals: return (None, None)
            a = np.array(vals, float); return (float(a.mean()), float(a.std()))
        agg = {
            "dataset": dataset_tag, "depth": D, "seeds": seeds, "n": len(per_seed),
            "test_acc": ms("test_acc"), "best_val": ms("best_val"),
            "peak_mem_mb": ms("peak_mem_mb"), "s_per_gen": ms("s_per_gen"),
            "gen_to_85": ms("gen_to_85"), "gen_to_90": ms("gen_to_90"),
            "n_trained": sum(r["trained_at_all"] for r in per_seed),
            "n_params": per_seed[0]["architecture"]["n_params"],
        }
        with open(os.path.join(out_dir, f"e1_agg_{dataset_tag}_D{D}.json"), "w") as f:
            json.dump(agg, f, indent=2)
        m, s = agg["best_val"]
        print(f"### [{dataset_tag}] D={D}: best_val {m:.2f}±{s:.2f} "
              f"| {agg['n_trained']}/{len(per_seed)} seeds trained "
              f"| params {agg['n_params']}")

    # write the depth-curve summary (the anchor figure's data)
    curve = {}
    for D in depths:
        ds_recs = [r for r in all_records if r["depth"] == D]
        bvs = np.array([r["best_val"] for r in ds_recs], float)
        curve[str(D)] = {
            "best_val_mean": float(bvs.mean()), "best_val_std": float(bvs.std()),
            "n_trained": int(sum(r["trained_at_all"] for r in ds_recs)),
            "n_params": ds_recs[0]["architecture"]["n_params"],
        }
    with open(os.path.join(out_dir, f"e1_depth_curve_{dataset_tag}.json"), "w") as f:
        json.dump(curve, f, indent=2)
    print(f"\n### [{dataset_tag}] depth curve written -> e1_depth_curve_{dataset_tag}.json")
    return all_records
