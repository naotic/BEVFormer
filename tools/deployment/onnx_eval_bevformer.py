#!/usr/bin/env python3
"""Run BEVFormer ONNX on nuScenes val samples and decode detections.

This is the deployment-side eval driver: preprocessing comes from the existing
val pipeline, unsupported training kernels are replaced only through the export
module patches, and bbox decoding remains in the host code just like deployment.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('BEVFORMER_STOCK_MMCV', '1')
for _k in (
        'ENABLE_FROZEN_BN', 'ENABLE_CONV_EPILOGUE', 'ENABLE_CONV_BN_FUSE',
        'ENABLE_DCN_GEMM_BN_RELU', 'ENABLE_FFN_GEMM_RELU',
        'MSDA_COL2IM_SMEM_REDUCE', 'ENABLE_BACKBONE_CL', 'DCN_RECOMPUTE_COLS'):
    os.environ.setdefault(_k, '0')
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')

import numpy as np
import onnxruntime as ort
import torch
from mmcv import Config
from mmdet3d.core import bbox3d2result
from projects.mmdet3d_plugin.datasets import custom_build_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / 'tools')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location(
    'pytorch2onnx_bevformer', REPO_ROOT / 'tools/deployment/pytorch2onnx_bevformer.py')
exp = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules[_spec.name] = exp
_spec.loader.exec_module(exp)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('onnx')
    p.add_argument('--data-root', default='./data/nuscenes')
    p.add_argument('--device', default='musa:0')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--max-samples', type=int, default=10,
                   help='number of samples; use -1 to run from --start to dataset end')
    p.add_argument('--save-every', type=int, default=20,
                   help='periodically write partial result pkl/summary during long runs')
    p.add_argument('--ort-threads', type=int, default=32)
    p.add_argument('--ort-opt-level', choices=['disable', 'basic', 'extended', 'all'],
                   default='disable', help='ONNX Runtime graph optimization level')
    p.add_argument('--ort-provider', choices=['auto', 'cuda', 'cpu'], default='auto',
                   help='ONNX Runtime execution provider selection')
    p.add_argument('--compare-pytorch', action='store_true')
    p.add_argument('--json-prefix', default='work_dirs/bevformer_onnx/onnx_subset')
    p.add_argument('--official-eval', action='store_true',
                   help='call dataset.evaluate; intended for full val or experimental sliced dataset')
    return p.parse_args()


def ort_graph_optimization_level(name):
    levels = {
        'disable': ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        'basic': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        'extended': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        'all': ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    return levels[name]


def ort_providers(name):
    available = ort.get_available_providers()
    if name == 'cpu':
        return ['CPUExecutionProvider']
    if name == 'cuda':
        if 'CUDAExecutionProvider' not in available:
            raise RuntimeError(f'CUDAExecutionProvider is not available: {available}')
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if 'CUDAExecutionProvider' in available:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']


def unwrap(dc):
    return dc.data if hasattr(dc, 'data') else dc


def load_dataset(cfg, data_root):
    test_cfg = copy.deepcopy(cfg.data.test)
    test_cfg.data_root = data_root
    test_cfg.ann_file = os.path.join(data_root, 'nuscenes_infos_temporal_val.pkl')
    test_cfg.test_mode = True
    test_cfg.pop('samples_per_gpu', None)
    return custom_build_dataset(test_cfg)


def prepare_sample(dataset, idx, device):
    item = dataset[idx]
    img = unwrap(item['img'][0])
    meta = unwrap(item['img_metas'][0])
    if isinstance(meta, list):
        meta = meta[0]
    img = img.unsqueeze(0).to(device=device, dtype=torch.float32)
    can_bus = torch.as_tensor(meta['can_bus'], device=device, dtype=torch.float32).unsqueeze(0)
    lidar2img = torch.as_tensor(np.asarray(meta['lidar2img']), device=device, dtype=torch.float32).unsqueeze(0)
    return img, can_bus, lidar2img, meta


def update_temporal(meta, prev_info, can_bus):
    scene = meta.get('scene_token')
    raw_pos = can_bus[:, :3].clone()
    raw_angle = can_bus[:, -1:].clone()
    if scene != prev_info.get('scene_token'):
        prev_info['prev_bev'] = None
        prev_info['scene_token'] = scene
        can_bus[:, :3] = 0
        can_bus[:, -1] = 0
    else:
        can_bus[:, :3] = can_bus[:, :3] - prev_info['prev_pos']
        can_bus[:, -1] = can_bus[:, -1] - prev_info['prev_angle']
    prev_info['prev_pos'] = raw_pos
    prev_info['prev_angle'] = raw_angle
    return can_bus


def raw_to_results(head, raw_out, img_metas):
    pred = {
        'bev_embed': raw_out[0],
        'all_cls_scores': raw_out[1],
        'all_bbox_preds': raw_out[2],
        'enc_cls_scores': None,
        'enc_bbox_preds': None,
    }
    bbox_list = head.get_bboxes(pred, img_metas, rescale=False)
    return [dict(pts_bbox=bbox3d2result(b, s, l)) for b, s, l in bbox_list]


def stats(a, b):
    a = a.detach().float().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a, dtype=np.float32)
    b = b.detach().float().cpu().numpy() if isinstance(b, torch.Tensor) else np.asarray(b, dtype=np.float32)
    d = np.abs(a - b)
    if d.size == 0:
        return dict(max_abs=0.0, mean_abs=0.0, p95_abs=0.0)
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()), p95_abs=float(np.percentile(d.reshape(-1), 95)))


def compare_boxes(pt, ot, topk=50):
    pb, ps, pl = pt['pts_bbox']['boxes_3d'].tensor, pt['pts_bbox']['scores_3d'], pt['pts_bbox']['labels_3d']
    ob, os_, ol = ot['pts_bbox']['boxes_3d'].tensor, ot['pts_bbox']['scores_3d'], ot['pts_bbox']['labels_3d']
    n = min(topk, pb.shape[0], ob.shape[0])
    return {
        'topk': int(n),
        'num_pt': int(pb.shape[0]),
        'num_onnx': int(ob.shape[0]),
        'bbox': stats(pb[:n], ob[:n]),
        'scores': stats(ps[:n], os_[:n]),
        'labels_equal': int((pl[:n] == ol[:n]).sum().item()) if n else 0,
    }


def main():
    args = parse_args()
    t0 = time.time()
    device = exp.resolve_device(args.device)
    cfg = Config.fromfile(args.config)
    dataset = load_dataset(cfg, args.data_root)
    report = exp.Report()
    checkpoint = None if str(args.checkpoint).lower() in ("none", "null", "skip") else args.checkpoint
    cfg_model, model = exp.build_model(args.config, checkpoint, device, report)
    wrapper = exp.BEVFormerOnnxWrapper(model, 928, 1600, 6).eval()
    head = model.pts_bbox_head

    so = ort.SessionOptions()
    so.graph_optimization_level = ort_graph_optimization_level(args.ort_opt_level)
    if args.ort_threads > 0:
        so.intra_op_num_threads = args.ort_threads
    sess = ort.InferenceSession(args.onnx, so, providers=ort_providers(args.ort_provider))
    input_names = [i.name for i in sess.get_inputs()]

    end = len(dataset) if args.max_samples < 0 else min(len(dataset), args.start + args.max_samples)
    prev = {'scene_token': None, 'prev_bev': None, 'prev_pos': None, 'prev_angle': None}
    pt_prev = {'scene_token': None, 'prev_bev': None, 'prev_pos': None, 'prev_angle': None}
    onnx_results = []
    pt_results = []
    comparisons = []

    for n, idx in enumerate(range(args.start, end), 1):
        st = time.time()
        img, can_bus, lidar2img, meta = prepare_sample(dataset, idx, device)
        can_bus = update_temporal(meta, prev, can_bus)
        if prev['prev_bev'] is None:
            prev_bev = torch.zeros(1, cfg_model.model.pts_bbox_head.bev_h * cfg_model.model.pts_bbox_head.bev_w,
                                   cfg_model.model.pts_bbox_head.in_channels, device=device, dtype=torch.float32)
        else:
            prev_bev = prev['prev_bev']
        feed_all = {
            'img': img.detach().cpu().numpy(),
            'prev_bev': prev_bev.detach().cpu().numpy(),
            'can_bus': can_bus.detach().cpu().numpy(),
            'lidar2img': lidar2img.detach().cpu().numpy(),
        }
        ort_np = sess.run(None, {name: feed_all[name] for name in input_names})
        ort_t = [torch.from_numpy(x).to(device) for x in ort_np]
        prev['prev_bev'] = ort_t[0].permute(1, 0, 2).contiguous()
        onnx_det = raw_to_results(head, ort_t, [meta])[0]
        onnx_results.append(onnx_det)

        if args.compare_pytorch:
            img2, can2, l2i2, meta2 = prepare_sample(dataset, idx, device)
            can2 = update_temporal(meta2, pt_prev, can2)
            if pt_prev['prev_bev'] is None:
                pt_prev_bev = torch.zeros_like(prev_bev)
            else:
                pt_prev_bev = pt_prev['prev_bev']
            with torch.no_grad():
                pt_raw = wrapper(img2, pt_prev_bev, can2, l2i2)
            pt_prev['prev_bev'] = pt_raw[0].permute(1, 0, 2).contiguous()
            pt_det = raw_to_results(head, pt_raw, [meta2])[0]
            pt_results.append(pt_det)
            comparisons.append({
                'idx': idx,
                'sample_idx': meta.get('sample_idx'),
                'raw': {name: stats(a, b) for name, a, b in zip(
                    ['bev_embed', 'all_cls_scores', 'all_bbox_preds'], pt_raw, ort_t)},
                'post': compare_boxes(pt_det, onnx_det),
            })
        print(f'[{n}/{end-args.start}] idx={idx} sample={meta.get("sample_idx")} elapsed={time.time()-st:.1f}s')
        if args.save_every > 0 and (n % args.save_every == 0):
            prefix = Path(args.json_prefix)
            prefix.parent.mkdir(parents=True, exist_ok=True)
            with open(str(prefix) + '.onnx_results.pkl', 'wb') as f:
                pickle.dump(onnx_results, f)
            partial = {
                'onnx': args.onnx,
                'start': args.start,
                'num_samples': len(onnx_results),
                'target_end': end,
                'elapsed_sec': time.time() - t0,
                'partial': True,
            }
            Path(str(prefix) + '.summary.partial.json').write_text(json.dumps(partial, indent=2) + '\n')

    prefix = Path(args.json_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(str(prefix) + '.onnx_results.pkl', 'wb') as f:
        pickle.dump(onnx_results, f)
    if pt_results:
        with open(str(prefix) + '.pytorch_results.pkl', 'wb') as f:
            pickle.dump(pt_results, f)
    summary = {
        'onnx': args.onnx,
        'start': args.start,
        'num_samples': len(onnx_results),
        'elapsed_sec': time.time() - t0,
        'ort_threads': args.ort_threads,
        'ort_opt_level': args.ort_opt_level,
        'ort_provider': args.ort_provider,
        'comparisons': comparisons,
    }
    if comparisons:
        for key in ['bev_embed', 'all_cls_scores', 'all_bbox_preds']:
            vals = [c['raw'][key]['mean_abs'] for c in comparisons]
            summary[f'{key}_mean_abs_avg'] = float(np.mean(vals))
        summary['post_top50_label_equal_avg'] = float(np.mean([
            c['post']['labels_equal'] / max(c['post']['topk'], 1) for c in comparisons]))
        summary['post_score_mean_abs_avg'] = float(np.mean([
            c['post']['scores']['mean_abs'] for c in comparisons]))
    if args.official_eval:
        eval_dataset = dataset
        if len(onnx_results) != len(dataset):
            eval_dataset = copy.copy(dataset)
            eval_dataset.data_infos = dataset.data_infos[args.start:end]
        summary['official_eval'] = eval_dataset.evaluate(
            onnx_results, jsonfile_prefix=str(prefix) + '_nusc')
    Path(str(prefix) + '.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2)[:6000])


if __name__ == '__main__':
    main()
