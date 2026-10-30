#!/usr/bin/env python3
"""Run resumable BEVFormer ONNX eval shards.

This wrapper intentionally delegates one shard at a time to
onnx_eval_bevformer.py. If a shard summary already exists with the expected
sample count, it is skipped, so the command can be re-run after interruption.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_ENV = {
    "PYTHONPATH": ".",
    "BEVFORMER_STOCK_MMCV": "1",
    "ENABLE_FROZEN_BN": "0",
    "ENABLE_CONV_EPILOGUE": "0",
    "ENABLE_CONV_BN_FUSE": "0",
    "ENABLE_DCN_GEMM_BN_RELU": "0",
    "ENABLE_FFN_GEMM_RELU": "0",
    "MSDA_OPT_ENABLE": "0",
    "MSDA_COL2IM_SMEM_REDUCE": "0",
    "ENABLE_BACKBONE_CL": "0",
    "TORCHDYNAMO_DISABLE": "1",
    "TORCH_COMPILE_DISABLE": "1",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("onnx")
    p.add_argument("--data-root", default="./data/nuscenes")
    p.add_argument("--device", default="musa:0")
    p.add_argument("--dataset-len", type=int, default=6019)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None,
                   help="exclusive end index; default is dataset-len")
    p.add_argument("--shard-size", type=int, default=100)
    p.add_argument("--max-shards", type=int, default=-1,
                   help="limit number of shards this invocation runs")
    p.add_argument("--ort-threads", type=int, default=64)
    p.add_argument("--ort-opt-level", choices=["disable", "basic", "extended", "all"],
                   default="all")
    p.add_argument("--ort-provider", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--out-dir", default="work_dirs/bevformer_onnx/full_shards")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def expected_count(start, end):
    return max(0, end - start)


def completed(prefix, count):
    summary = Path(str(prefix) + ".summary.json")
    pkl = Path(str(prefix) + ".onnx_results.pkl")
    if not summary.exists() or not pkl.exists():
        return False
    try:
        meta = json.loads(summary.read_text())
    except Exception:
        return False
    return int(meta.get("start", -1)) == int(prefix.name.split("_s", 1)[1].split("_n", 1)[0]) \
        and int(meta.get("num_samples", -1)) == count


def main():
    args = parse_args()
    end = args.dataset_len if args.end is None else min(args.end, args.dataset_len)
    if args.shard_size <= 0:
        raise SystemExit("--shard-size must be positive")
    if not (0 <= args.start <= end <= args.dataset_len):
        raise SystemExit(f"invalid range: start={args.start}, end={end}, dataset_len={args.dataset_len}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(DEFAULT_ENV)

    total_shards = math.ceil((end - args.start) / args.shard_size)
    ran = 0
    skipped = 0
    failed = 0
    t0 = time.time()
    for shard_idx, s in enumerate(range(args.start, end, args.shard_size), 1):
        e = min(s + args.shard_size, end)
        n = expected_count(s, e)
        prefix = out_dir / f"shard_s{s}_n{n}"
        log_path = Path(str(prefix) + ".log")
        if not args.force and completed(prefix, n):
            skipped += 1
            print(f"[skip] {prefix.name} already complete ({n} samples)", flush=True)
            continue
        if args.max_shards >= 0 and ran >= args.max_shards:
            print(f"[stop] reached --max-shards={args.max_shards}", flush=True)
            break

        cmd = [
            sys.executable,
            "tools/deployment/onnx_eval_bevformer.py",
            args.config,
            args.checkpoint,
            args.onnx,
            "--data-root", args.data_root,
            "--device", args.device,
            "--start", str(s),
            "--max-samples", str(n),
            "--ort-threads", str(args.ort_threads),
            "--ort-opt-level", args.ort_opt_level,
            "--ort-provider", args.ort_provider,
            "--save-every", str(args.save_every),
            "--json-prefix", str(prefix),
        ]
        print(f"[run] shard {shard_idx}/{total_shards}: {prefix.name}", flush=True)
        print("      " + " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        with open(log_path, "w") as log:
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            failed += 1
            print(f"[fail] {prefix.name}; see {log_path}", flush=True)
            break
        ran += 1
        print(f"[done] {prefix.name}; elapsed_total={time.time() - t0:.1f}s", flush=True)

    summary = {
        "range": [args.start, end],
        "shard_size": args.shard_size,
        "ran": ran,
        "skipped": skipped,
        "failed": failed,
        "elapsed_sec": time.time() - t0,
        "out_dir": str(out_dir),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
