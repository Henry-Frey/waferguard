"""WaferGuard training optimization profiler.

Benchmarks data loading and GPU compute across both RTX 3090s, finds the
largest batch size that keeps each card below a VRAM ceiling, and writes
configs/dual_gpu.yaml with recommended settings.

Usage
-----
    # Full profile (needs data at the path in configs/sota.yaml)
    python scripts/profile_training.py --config=configs/sota.yaml

    # Skip data loading section (no dataset required)
    python scripts/profile_training.py --config=configs/sota.yaml --no-data

    # Only use GPU 0 — useful if GPU 1 is occupied
    python scripts/profile_training.py --config=configs/sota.yaml --single-gpu
"""
from __future__ import annotations

import argparse
import sys
import time
import textwrap
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.models.classifier import build_classifier

# ── tunables ──────────────────────────────────────────────────────────────────
TARGET_VRAM_FRAC = 0.70   # keep each GPU below 70 % — comfortable headroom
WARMUP_STEPS    = 5
BENCH_STEPS     = 20
# ──────────────────────────────────────────────────────────────────────────────


# ── helpers ───────────────────────────────────────────────────────────────────

def gpu_info() -> list[dict]:
    out = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        out.append({"id": i, "name": p.name, "total_mb": p.total_memory / 1024 ** 2})
    return out


def peak_mb(device_id: int = 0) -> float:
    return torch.cuda.max_memory_allocated(device_id) / 1024 ** 2


def reset_peak(device_id: int = 0) -> None:
    torch.cuda.reset_peak_memory_stats(device_id)


def hr(char: str = "─", width: int = 70) -> None:
    print(char * width)


# ── data loading benchmark ────────────────────────────────────────────────────

def _loader_for(cfg, nw: int, n_samples: int = 2000):
    """Build a loader using first n_samples of the dataset."""
    from src.data.wm811k import WM811KDataset, WM811KDatasetV2, preprocess_wm811k
    from src.data.transforms import build_classifier_transforms
    from torch.utils.data import DataLoader

    print(f"  Loading {n_samples} samples for loader benchmark …", flush=True)
    maps, labels = preprocess_wm811k(
        cfg.wm811k.path,
        img_size=cfg.data.img_size,
        noise_filter=False,   # skip denoising so load time isn't biased
    )
    n = min(n_samples, len(labels))
    maps, labels = maps[:n], labels[:n]

    fc = cfg.get("features", {})
    tfms = build_classifier_transforms(cfg.data.img_size, is_train=True)
    if fc.get("use_radon") or fc.get("use_distance_from_center"):
        ds = WM811KDatasetV2(
            maps, labels, cfg.data.img_size, tfms,
            use_radon=fc.get("use_radon", False),
            use_distance=fc.get("use_distance_from_center", False),
            radon_angles=fc.get("radon_angles", 36),
        )
    else:
        ds = WM811KDataset(maps, labels, cfg.data.img_size, tfms)

    return DataLoader(
        ds, batch_size=cfg.classifier.batch_size,
        num_workers=nw, pin_memory=True, drop_last=True,
    )


def bench_loader(cfg) -> dict[int, dict]:
    """Sweep num_workers=[0,2,4] and return throughput stats."""
    results = {}
    for nw in [0, 2, 4]:
        print(f"\n  Benchmarking DataLoader  num_workers={nw} …", flush=True)
        loader = _loader_for(cfg, nw)
        n_batches = min(25, len(loader))
        t0 = time.perf_counter()
        total = 0
        for i, batch in enumerate(loader):
            total += batch["image"].shape[0]
            if i + 1 >= n_batches:
                break
        elapsed = time.perf_counter() - t0
        stats = {
            "images_per_sec": total / elapsed,
            "ms_per_batch":   elapsed * 1000 / n_batches,
        }
        results[nw] = stats
        print(f"    → {stats['images_per_sec']:>6.0f} img/s   "
              f"{stats['ms_per_batch']:.1f} ms/batch")
    return results


# ── compute benchmark ─────────────────────────────────────────────────────────

def _step(model, criterion, scaler, optimizer, x, y):
    """One AMP forward+backward step."""
    with autocast("cuda"):
        loss = criterion(model(x), y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()


def bench_step(
    cfg,
    batch_size: int,
    devices: list[torch.device],
    use_data_parallel: bool = False,
    n_steps: int = BENCH_STEPS,
) -> dict:
    """Measure fwd+bwd throughput and peak VRAM."""
    model = build_classifier(cfg)
    if use_data_parallel and len(devices) > 1:
        model = nn.DataParallel(model, device_ids=[d.index for d in devices])
    model = model.to(devices[0]).train()

    in_ch = 3 + sum([
        cfg.get("features", {}).get("use_radon", False),
        cfg.get("features", {}).get("use_distance_from_center", False),
    ])
    x = torch.randn(batch_size, in_ch, cfg.data.img_size, cfg.data.img_size, device=devices[0])
    y = torch.randint(0, cfg.wm811k.num_classes, (batch_size,), device=devices[0])

    scaler    = GradScaler("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss().to(devices[0])

    # warmup
    for _ in range(WARMUP_STEPS):
        _step(model, criterion, scaler, optimizer, x, y)

    for d in devices:
        reset_peak(d.index)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        _step(model, criterion, scaler, optimizer, x, y)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_mb = torch.cuda.get_device_properties(devices[0].index).total_memory / 1024 ** 2
    result = {
        "ms_per_step":    elapsed * 1000 / n_steps,
        "samples_per_sec": batch_size * n_steps / elapsed,
        "peak_mb_gpu0":   peak_mb(devices[0].index),
        "vram_frac_gpu0": peak_mb(devices[0].index) / total_mb,
    }
    if use_data_parallel and len(devices) > 1:
        result["peak_mb_gpu1"]   = peak_mb(devices[1].index)
        result["vram_frac_gpu1"] = peak_mb(devices[1].index) / total_mb

    del model, x, y, scaler, optimizer
    torch.cuda.empty_cache()
    return result


def find_max_batch(cfg, device: torch.device, target_frac: float = TARGET_VRAM_FRAC) -> int:
    """Binary search: largest batch_size that stays below target VRAM fraction."""
    total_mb  = torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 2
    target_mb = total_mb * target_frac
    lo, hi = 32, 640
    best = lo
    in_ch = 3 + sum([
        cfg.get("features", {}).get("use_radon", False),
        cfg.get("features", {}).get("use_distance_from_center", False),
    ])
    while lo <= hi:
        mid = ((lo + hi) // 2 // 16) * 16   # round to multiple of 16
        if mid <= 0:
            break
        try:
            torch.cuda.empty_cache()
            model = build_classifier(cfg).to(device).train()
            x = torch.randn(mid, in_ch, cfg.data.img_size, cfg.data.img_size, device=device)
            y = torch.randint(0, cfg.wm811k.num_classes, (mid,), device=device)
            scaler = GradScaler("cuda")
            opt    = torch.optim.AdamW(model.parameters(), lr=3e-4)
            crit   = nn.CrossEntropyLoss().to(device)
            for _ in range(3):
                _step(model, crit, scaler, opt, x, y)
            used = peak_mb(device.index)
            del model, x, y, scaler, opt
            torch.cuda.empty_cache()

            if used <= target_mb:
                best = mid
                lo   = mid + 16
                print(f"    batch={mid:>4}  VRAM {used:.0f}/{total_mb:.0f} MB "
                      f"({used/total_mb*100:.0f}%)  ✓")
            else:
                hi = mid - 16
                print(f"    batch={mid:>4}  VRAM {used:.0f}/{total_mb:.0f} MB "
                      f"({used/total_mb*100:.0f}%)  ✗ (over {target_frac*100:.0f}% limit)")
        except torch.cuda.OutOfMemoryError:
            hi = mid - 16
            torch.cuda.empty_cache()
            print(f"    batch={mid:>4}  OOM")
    return best


# ── config writer ─────────────────────────────────────────────────────────────

def write_dual_gpu_yaml(
    sota_path: str,
    optimal_bs: int,
    optimal_workers: int,
) -> Path:
    """Write configs/dual_gpu.yaml — dual-GPU config derived from sota.yaml."""
    from src.utils.config import load_config
    from omegaconf import OmegaConf

    cfg = load_config(sota_path)
    cfg.data.num_workers        = optimal_workers
    cfg.classifier.batch_size   = optimal_bs
    cfg.project_name            = "waferguard-dual-gpu"
    if not OmegaConf.is_struct(cfg):
        OmegaConf.set_struct(cfg, False)
    # DataParallel flag read by the trainer
    if not hasattr(cfg, "training"):
        cfg.training = OmegaConf.create({})
    cfg.training.data_parallel = True
    cfg.training.gpu_ids       = [0, 1]

    out = ROOT / "configs" / "dual_gpu.yaml"
    OmegaConf.save(cfg, str(out))
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="WaferGuard training profiler")
    parser.add_argument("--config",     default="configs/sota.yaml")
    parser.add_argument("--no-data",    action="store_true",
                        help="skip DataLoader benchmark (no dataset required)")
    parser.add_argument("--single-gpu", action="store_true",
                        help="benchmark GPU 0 only (GPU 1 stays idle)")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    gpus   = gpu_info()
    n_gpu  = 1 if args.single_gpu else min(len(gpus), 2)
    devs   = [torch.device(f"cuda:{i}") for i in range(n_gpu)]

    # ── 1. GPU info ───────────────────────────────────────────────────────────
    hr("═")
    print("  WaferGuard Training Profiler")
    hr("═")
    print(f"\n  Available GPUs ({len(gpus)} found, profiling {n_gpu}):")
    for g in gpus:
        marker = "  (profiling)" if g["id"] < n_gpu else "  (idle)"
        print(f"    GPU {g['id']}: {g['name']}  {g['total_mb']:.0f} MB{marker}")
    print(f"\n  VRAM ceiling: {TARGET_VRAM_FRAC*100:.0f}% per card "
          f"= {gpus[0]['total_mb'] * TARGET_VRAM_FRAC:.0f} MB\n")

    # ── 2. DataLoader benchmark ───────────────────────────────────────────────
    loader_results = {}
    best_workers   = cfg.data.num_workers

    if not args.no_data:
        hr()
        print("  Section 1 — DataLoader throughput (CPU)")
        hr()
        try:
            loader_results = bench_loader(cfg)
            # Pick the num_workers with the highest throughput
            best_workers = max(loader_results, key=lambda nw: loader_results[nw]["images_per_sec"])
            print(f"\n  Best num_workers: {best_workers}  "
                  f"({loader_results[best_workers]['images_per_sec']:.0f} img/s)")

            # warn if the loader is the bottleneck
            loader_ms   = loader_results[best_workers]["ms_per_batch"]
            print(f"  DataLoader ms/batch: {loader_ms:.1f} ms")
            print("  Note: if this is > single-GPU step time, "
                  "the loader is the bottleneck → increase num_workers.")
        except Exception as e:
            print(f"  Skipped (data not found or error): {e}")
            print("  Re-run without --no-data once the dataset is in place.")
    else:
        print("\n  Section 1 — DataLoader benchmark skipped (--no-data)\n")

    # ── 3. Single-GPU step benchmark ─────────────────────────────────────────
    hr()
    print("  Section 2 — Single-GPU step benchmark (GPU 0)")
    hr()

    current_bs = cfg.classifier.batch_size
    print(f"\n  Current config batch_size = {current_bs}")

    print(f"  Benchmarking batch={current_bs} on GPU 0 …", flush=True)
    single = bench_step(cfg, current_bs, [devs[0]])
    print(f"    {single['ms_per_step']:.1f} ms/step  "
          f"{single['samples_per_sec']:.0f} img/s  "
          f"VRAM {single['peak_mb_gpu0']:.0f} MB "
          f"({single['vram_frac_gpu0']*100:.0f}%)")

    # ── 4. Find optimal single-GPU batch size ─────────────────────────────────
    hr()
    print(f"  Section 3 — Find max batch size at {TARGET_VRAM_FRAC*100:.0f}% VRAM (GPU 0)")
    hr()
    print()
    optimal_single = find_max_batch(cfg, devs[0], target_frac=TARGET_VRAM_FRAC)
    print(f"\n  Max single-GPU batch at {TARGET_VRAM_FRAC*100:.0f}% VRAM: {optimal_single}")

    # ── 5. DataParallel benchmark ─────────────────────────────────────────────
    optimal_dp = optimal_single
    dp_speedup  = 1.0

    if n_gpu >= 2:
        hr()
        print("  Section 4 — DataParallel benchmark (GPU 0 + GPU 1)")
        hr()

        # For DP, each GPU gets batch/2, so total batch = 2× single-GPU optimal
        dp_bs = (optimal_single // 2) * 2   # must be even for DP split
        print(f"\n  DataParallel batch_size = {dp_bs}  "
              f"({dp_bs//2} per GPU)\n")

        dp = bench_step(cfg, dp_bs, devs, use_data_parallel=True)
        print(f"  DataParallel: {dp['ms_per_step']:.1f} ms/step  "
              f"{dp['samples_per_sec']:.0f} img/s  "
              f"GPU0 {dp['peak_mb_gpu0']:.0f} MB ({dp['vram_frac_gpu0']*100:.0f}%)  "
              f"GPU1 {dp['peak_mb_gpu1']:.0f} MB ({dp['vram_frac_gpu1']*100:.0f}%)")
        print(f"  Single-GPU:   {single['ms_per_step']:.1f} ms/step  "
              f"{single['samples_per_sec']:.0f} img/s\n")

        dp_speedup = dp["samples_per_sec"] / single["samples_per_sec"]
        print(f"  DataParallel speedup: {dp_speedup:.2f}×")
        if dp_speedup < 1.3:
            print("  ⚠  Speedup < 1.3× — DataParallel overhead is high for this")
            print("     model/batch size. Consider DDP (DistributedDataParallel)")
            print("     or simply running two separate experiments in parallel.")
        else:
            print("  ✓  Worthwhile speedup — dual_gpu.yaml will enable DataParallel.")

        # Find optimal DP batch (both GPUs at 70 %)
        print(f"\n  Finding optimal DataParallel batch at {TARGET_VRAM_FRAC*100:.0f}% per GPU …")
        # DP optimal: each GPU handles batch/2, so target batch = 2 × single optimal
        optimal_dp = min(optimal_single * 2, 512)   # 512 is a safe cap
        print(f"  Recommended DP batch_size: {optimal_dp}")

    # ── 6. Recommendations ───────────────────────────────────────────────────
    hr("═")
    print("  Recommendations")
    hr("═")
    final_bs      = optimal_dp if n_gpu >= 2 else optimal_single
    final_workers = best_workers

    print(f"""
  Config           : {args.config}
  GPUs used        : {n_gpu}
  Recommended batch: {final_bs}  (current: {current_bs})
  Recommended workers: {final_workers}  (current: {cfg.data.num_workers})
  SAM interval     : {cfg.classifier.optimizer.get('sam_interval', 1)}
  Expected VRAM    : ~{TARGET_VRAM_FRAC*100:.0f}% per card

  SAM doubles the per-step cost (two fwd+bwd passes).
  With sam_interval=10, overhead is ~10% — recommended as-is.

  If the DataLoader is the bottleneck (ms/batch > ms/step):
    → Increase num_workers to 4 or precompute Radon channels
      offline (store as .npy alongside LSWMD.pkl).

  If you want faster iteration during experiments:
    → Disable Radon (features.use_radon=false) for a 2–3× loader speedup.
    → Re-enable for final training runs.
""")

    # ── 7. Write dual_gpu.yaml ────────────────────────────────────────────────
    if n_gpu >= 2 and dp_speedup >= 1.2:
        out = write_dual_gpu_yaml(args.config, final_bs, final_workers)
        print(f"  Written: {out.relative_to(ROOT)}")
        print(f"  Train:   python -m src.training.train_classifier --config={out.relative_to(ROOT)}")
    else:
        print("  dual_gpu.yaml not written — single-GPU is recommended for this setup.")
        print(f"  Use: python -m src.training.train_classifier --config={args.config} "
              f"classifier.batch_size={final_bs} data.num_workers={final_workers}")

    hr("═")


if __name__ == "__main__":
    main()
