#!/usr/bin/env python3
"""Merge BEVFormer ONNX eval shards and optionally run official nuScenes eval."""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import pickle
from pathlib import Path

from mmcv import Config
from projects.mmdet3d_plugin.datasets import custom_build_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--data-root", default="./data/nuscenes")
    p.add_argument("--chunks", nargs="+", required=True,
                   help="summary json files, pkl files, or glob patterns for shard outputs")
    p.add_argument("--out-prefix", default="work_dirs/bevformer_onnx/merged_full")
    p.add_argument("--official-eval", action="store_true")
    return p.parse_args()


def load_dataset(cfg, data_root):
    test_cfg = copy.deepcopy(cfg.data.test)
    test_cfg.data_root = data_root
    test_cfg.ann_file = os.path.join(data_root, "nuscenes_infos_temporal_val.pkl")
    test_cfg.test_mode = True
    test_cfg.pop("samples_per_gpu", None)
    return custom_build_dataset(test_cfg)


def expand(paths):
    out = []
    for item in paths:
        matches = sorted(glob.glob(item))
        out.extend(matches if matches else [item])
    return out


def summary_for(path):
    p = Path(path)
    if p.name.endswith(".summary.json"):
        return p
    if p.name.endswith(".summary.partial.json"):
        return p
    if p.name.endswith(".onnx_results.pkl"):
        stem = str(p)[:-len(".onnx_results.pkl")]
        cand = Path(stem + ".summary.json")
        if cand.exists():
            return cand
        cand = Path(stem + ".summary.partial.json")
        if cand.exists():
            return cand
    raise FileNotFoundError(f"cannot find summary for {path}")


def pkl_for(summary):
    name = str(summary)
    if name.endswith(".summary.json"):
        return Path(name[:-len(".summary.json")] + ".onnx_results.pkl")
    if name.endswith(".summary.partial.json"):
        return Path(name[:-len(".summary.partial.json")] + ".onnx_results.pkl")
    raise ValueError(summary)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dataset = load_dataset(cfg, args.data_root)
    results = [None] * len(dataset)
    shards = []

    for item in expand(args.chunks):
        sp = summary_for(item)
        meta = json.loads(sp.read_text())
        pp = pkl_for(sp)
        with open(pp, "rb") as f:
            part = pickle.load(f)
        start = int(meta["start"])
        declared = int(meta.get("num_samples", len(part)))
        if declared != len(part):
            raise ValueError(f"{pp}: summary num_samples={declared}, pkl len={len(part)}")
        for off, res in enumerate(part):
            idx = start + off
            if idx >= len(results):
                raise IndexError(f"{pp}: index {idx} outside dataset length {len(results)}")
            if results[idx] is not None:
                raise ValueError(f"duplicate prediction for dataset index {idx}")
            results[idx] = res
        shards.append({"summary": str(sp), "pkl": str(pp), "start": start, "num_samples": len(part)})

    missing = [i for i, r in enumerate(results) if r is None]
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset_len": len(dataset),
        "num_predictions": len(results) - len(missing),
        "num_missing": len(missing),
        "missing_first20": missing[:20],
        "shards": shards,
    }
    if missing:
        Path(str(out_prefix) + ".merge_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        raise SystemExit(f"missing {len(missing)} predictions; first missing: {missing[:20]}")

    with open(str(out_prefix) + ".onnx_results.pkl", "wb") as f:
        pickle.dump(results, f)
    if args.official_eval:
        summary["official_eval"] = dataset.evaluate(
            results, jsonfile_prefix=str(out_prefix) + "_nusc")
    Path(str(out_prefix) + ".merge_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2)[:6000])


if __name__ == "__main__":
    main()
