#!/usr/bin/env python3
"""Export the accepted V4 Stage-1 network for Unity Sentis 2.6.

The graph emits only calibrated pole and line scores for the central 48^3 core.
This preserves the deployed score fusion while avoiding readback of unused heads
and the 8-voxel context border. FP16 is an optional candidate, never an automatic
replacement for FP32: real-patch decision parity is recorded in the sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from precision_common import fuse_scores
from v4_realtime_core import build_v4_patch, load_calibration, load_v4_model
from voxel_common import load_npz_sparse


INPUT_NAME = "volume"
OUTPUT_NAMES = ("pole_score", "line_score")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--stage2-profile", required=True)
    p.add_argument("--stage2-bundle", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--core-size", type=int, default=48)
    p.add_argument("--grid-size", type=int, nargs=3, default=(400, 400, 200))
    p.add_argument("--voxel-size-ft", type=float, default=.5)
    p.add_argument("--real-npz", action="append", default=[])
    p.add_argument("--max-real-patches", type=int, default=32)
    p.add_argument("--fp16", type=int, default=1)
    p.add_argument("--verify", type=int, default=1)
    p.add_argument("--trt-benchmark-runs", type=int, default=20)
    p.add_argument("--build-trt-plan", type=int, default=0)
    return p.parse_args()


class SentisStage1(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, calibration: dict, core_size: int):
        super().__init__()
        self.model = model
        self.sem_weight = float(calibration["score_sem_weight"])
        self.binary_weight = float(calibration["score_binary_weight"])
        self.object_weight = float(calibration["score_object_weight"])
        self.core_size = int(core_size)

    def forward(self, volume):
        out = self.model(volume)
        pole, line = fuse_scores(
            out, self.sem_weight, self.binary_weight, self.object_weight
        )
        patch = int(volume.shape[-1])
        pad = (patch - self.core_size) // 2
        end = pad + self.core_size
        return pole[:, pad:end, pad:end, pad:end], line[:, pad:end, pad:end, pad:end]


def export_fp32(wrapper: torch.nn.Module, target: Path, opset: int, batch: int, patch: int) -> None:
    dummy = torch.zeros((batch, 5, patch, patch, patch), dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy,
        str(target),
        input_names=[INPUT_NAME],
        output_names=list(OUTPUT_NAMES),
        dynamic_axes={INPUT_NAME: {0: "batch"}, **{name: {0: "batch"} for name in OUTPUT_NAMES}},
        opset_version=opset,
        do_constant_folding=True,
    )


def convert_fp16(source: Path, target: Path) -> None:
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(source))
    model = float16.convert_float_to_float16(
        model, keep_io_types=True, disable_shape_infer=False
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(target))


def onnx_metadata(path: Path) -> dict:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    ops = sorted({node.op_type for node in model.graph.node})
    return {
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "ir_version": int(model.ir_version),
        "opsets": {item.domain or "ai.onnx": int(item.version) for item in model.opset_import},
        "operators": ops,
    }


def synthetic_input(batch: int, patch: int) -> np.ndarray:
    rng = np.random.default_rng(20260911)
    x = np.zeros((batch, 5, patch, patch, patch), np.float32)
    x[:, 0] = (rng.random((batch, patch, patch, patch)) > .985).astype(np.float32)
    axis = np.arange(patch, dtype=np.float32)
    norm = (axis / max(patch - 1, 1)) * 2.0 - 1.0
    x[:, 1] = norm[None, None, None, :]
    x[:, 2] = norm[None, None, :, None]
    x[:, 3] = norm[None, :, None, None]
    x[:, 4] = rng.uniform(-1, 1, x[:, 4].shape).astype(np.float32) * x[:, 0]
    return x


def real_patches(args: argparse.Namespace, cfg: dict) -> list[np.ndarray]:
    patches = []
    patch = int(cfg.get("patch_size", 64))
    for filename in args.real_npz:
        item = load_npz_sparse(filename, tuple(args.grid_size), bool(int(cfg.get("use_dist", 1))))
        if not len(item["coords"]):
            continue
        keys = np.unique(item["coords"] // int(args.core_size), axis=0)
        keys = keys[np.lexsort((keys[:, 0], keys[:, 1], keys[:, 2]))]
        for key in keys:
            center = key.astype(np.int64) * int(args.core_size) + int(args.core_size) // 2
            patches.append(build_v4_patch(
                item, center, tuple(args.grid_size), patch,
                bool(int(cfg.get("use_coord_channels", 1))), bool(int(cfg.get("use_dist", 1)))
            ))
            if len(patches) >= args.max_real_patches:
                return patches
    return patches


def session(path: Path, providers) -> object:
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=providers)


def compare_model(
    path: Path, wrapper: torch.nn.Module, inputs: list[np.ndarray], calibration: dict,
    prefer_cuda: bool = False,
) -> dict:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if prefer_cuda and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess = session(path, providers)
    max_abs = 0.0
    total_values = 0
    label_mismatches = 0
    reference_positives = 0
    candidate_positives = 0
    for x in inputs:
        xb = x[None] if x.ndim == 4 else x
        with torch.inference_mode():
            ref = [q.detach().cpu().numpy() for q in wrapper(torch.from_numpy(xb))]
        got = sess.run(list(OUTPUT_NAMES), {INPUT_NAME: xb.astype(np.float32, copy=False)})
        for left, right in zip(ref, got):
            max_abs = max(max_abs, float(np.max(np.abs(left - right))))
            total_values += int(left.size)
        ref_label = labels(ref[0], ref[1], calibration)
        got_label = labels(got[0], got[1], calibration)
        label_mismatches += int(np.count_nonzero(ref_label != got_label))
        reference_positives += int(np.count_nonzero(ref_label))
        candidate_positives += int(np.count_nonzero(got_label))
    return {
        "patches": len(inputs),
        "values_compared": total_values,
        "max_abs_score_delta": max_abs,
        "label_mismatches": label_mismatches,
        "reference_positive_labels": reference_positives,
        "candidate_positive_labels": candidate_positives,
        "decision_parity": label_mismatches == 0,
        "providers": sess.get_providers(),
    }


def labels(pole: np.ndarray, line: np.ndarray, calibration: dict) -> np.ndarray:
    result = np.zeros(pole.shape, np.int8)
    pm = pole >= float(calibration["pole_threshold"])
    lm = line >= float(calibration["line_threshold"])
    result[pm & ~lm] = 1
    result[lm & ~pm] = 2
    both = pm & lm
    result[both & (pole >= line)] = 1
    result[both & (line > pole)] = 2
    return result


def benchmark(path: Path, inputs: np.ndarray, runs: int) -> dict:
    import onnxruntime as ort

    available = ort.get_available_providers()
    results = {"available_providers": available, "runs": runs, "providers": {}}
    candidates = [
        ("cpu", ["CPUExecutionProvider"]),
        ("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        ("tensorrt_fp16", [
            ("TensorrtExecutionProvider", {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(path.parent / "trt_engine_cache"),
            }),
            "CUDAExecutionProvider", "CPUExecutionProvider",
        ]),
    ]
    for name, providers in candidates:
        provider_name = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
        if provider_name not in available:
            results["providers"][name] = {"available": False}
            continue
        try:
            sess = session(path, providers)
            for _ in range(3):
                sess.run(list(OUTPUT_NAMES), {INPUT_NAME: inputs})
            started = time.perf_counter()
            for _ in range(runs):
                sess.run(list(OUTPUT_NAMES), {INPUT_NAME: inputs})
            results["providers"][name] = {
                "available": True,
                "mean_ms": (time.perf_counter() - started) * 1000.0 / runs,
                "actual_providers": sess.get_providers(),
            }
        except Exception as exc:
            results["providers"][name] = {"available": True, "error": repr(exc)}
    return results


def maybe_build_plan(path: Path, output: Path, enabled: bool) -> dict:
    if not enabled:
        return {"requested": False}
    executable = shutil.which("trtexec")
    if not executable:
        return {"requested": True, "built": False, "reason": "trtexec not found"}
    command = [
        executable, f"--onnx={path}", f"--saveEngine={output}", "--fp16",
        "--minShapes=volume:1x5x64x64x64", "--optShapes=volume:1x5x64x64x64",
        "--maxShapes=volume:12x5x64x64x64", "--skipInference",
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {
        "requested": True, "built": result.returncode == 0, "exit_code": result.returncode,
        "command": command, "log_tail": result.stdout[-8000:],
        "note": "benchmark-only TensorRT engine; Unity Sentis does not consume .plan files",
    }


def export_tree_bundle(bundle_path: Path, target: Path) -> dict:
    import joblib

    bundle = joblib.load(bundle_path)
    output = {
        "format_version": 1,
        "source_sha256": sha256(bundle_path),
        "feature_columns": list(bundle["feature_columns"]),
        "pole_threshold": float(bundle["pole_threshold"]),
        "line_threshold": float(bundle["line_threshold"]),
        "models": {},
    }
    for name in ("pole", "line"):
        model = bundle[name + "_model"]
        trees = []
        for estimator in model.estimators_:
            tree = estimator.tree_
            trees.append({
                "children_left": tree.children_left.astype(int).tolist(),
                "children_right": tree.children_right.astype(int).tolist(),
                "feature": tree.feature.astype(int).tolist(),
                "threshold": tree.threshold.astype(float).tolist(),
                "value": tree.value[:, 0, :].astype(float).tolist(),
            })
        output["models"][name] = {
            "classes": np.asarray(model.classes_).astype(int).tolist(),
            "trees": trees,
        }
    target.write_text(json.dumps(output, separators=(",", ":")) + "\n")
    return {"sha256": sha256(target), "bytes": target.stat().st_size}


def main() -> None:
    args = parse_args()
    if args.opset != 17:
        raise ValueError("Sentis release export is pinned to ONNX opset 17")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    calibration_path = Path(args.calibration).resolve()
    profile_path = Path(args.stage2_profile).resolve()
    bundle_path = Path(args.stage2_bundle).resolve()
    for path in (checkpoint, calibration_path, profile_path, bundle_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    model, cfg, _ = load_v4_model(str(checkpoint), "cpu", False)
    calibration = load_calibration(str(calibration_path))
    profile = json.loads(profile_path.read_text())
    patch = int(cfg.get("patch_size", 64))
    if patch != 64 or args.core_size != 48:
        raise RuntimeError("accepted deployment contract is patch=64, core=48")
    wrapper = SentisStage1(model, calibration, args.core_size).eval()
    fp32_path = output / "v10_stage1_voxelnet3d_fp32.onnx"
    fp16_path = output / "v10_stage1_voxelnet3d_fp16.onnx"
    export_fp32(wrapper, fp32_path, args.opset, args.batch_size, patch)
    if args.fp16:
        convert_fp16(fp32_path, fp16_path)

    inputs = [synthetic_input(1, patch)]
    real = real_patches(args, cfg)
    inputs.extend(real)
    verification = {}
    if args.verify:
        verification["fp32"] = compare_model(fp32_path, wrapper, inputs, calibration)
        if args.fp16:
            verification["fp16"] = compare_model(
                fp16_path, wrapper, inputs, calibration, prefer_cuda=True
            )
    fp16_approved = bool(
        args.fp16 and real and verification.get("fp16", {}).get("decision_parity", False)
    )

    tree_path = output / "v10_stage2_refiner_trees.json"
    tree_metadata = export_tree_bundle(bundle_path, tree_path)
    benchmark_input = synthetic_input(args.batch_size, patch)
    timing = benchmark(fp32_path, benchmark_input, args.trt_benchmark_runs) if args.trt_benchmark_runs > 0 else {}
    plan = maybe_build_plan(
        fp32_path, output / "benchmark_only_v10_stage1_fp16.plan", bool(args.build_trt_plan)
    )
    models = {"fp32": {"file": fp32_path.name, **onnx_metadata(fp32_path)}}
    if args.fp16:
        models["fp16"] = {"file": fp16_path.name, **onnx_metadata(fp16_path)}

    sidecar = {
        "format_version": 1,
        "task": "v10_stage1_inference_and_voxel_supported_stage2_reconstruction",
        "unity": {"version": "6.3 LTS", "sentis_package": "com.unity.ai.inference 2.6.1",
                  "backend": "BackendType.GPUCompute"},
        "onnx": {"opset": args.opset, "models": models,
                 "default_model": fp32_path.name,
                 "fp16_deployable": fp16_approved,
                 "fp16_rule": "use only when real-patch calibrated Stage1 label mismatches equal zero"},
        "input": {
            "name": INPUT_NAME, "layout": "NCDHW", "dtype": "float32",
            "shape": ["batch", 5, 64, 64, 64],
            "channels": ["occupancy", "local_x", "local_y", "local_z", "normalized_dist_center_ft"],
            "coordinate_formula": "clip((local_index/(grid_size-1))*2-1,-1.5,1.5)",
            "distance_formula": "per-slice dist_center_ft / max(max(abs(dist_center_ft)),1)",
            "grid_size_xyz": list(map(int, args.grid_size)),
        },
        "outputs": {
            "layout": "NDHW", "core_shape": [48, 48, 48],
            "names": list(OUTPUT_NAMES),
            "core_offset_in_patch": [8, 8, 8],
        },
        "calibration": {
            "pole_threshold": calibration["pole_threshold"],
            "line_threshold": calibration["line_threshold"],
            "score_weights": {
                "semantic": calibration["score_sem_weight"],
                "binary": calibration["score_binary_weight"],
                "objectness": calibration["score_object_weight"],
            },
            "source_file": calibration_path.name, "source_sha256": sha256(calibration_path),
        },
        "stage2": {
            "algorithm": "stage1-electrical-tracks-v10-voxel-supported-opt1-fix2-20260910",
            "profile": profile,
            "refiner_tree_file": tree_path.name,
            "refiner_tree_metadata": tree_metadata,
            "runtime_gt_usage": False, "synthetic_line_voxels": 0,
            "disconnected_fragment_bridges_allowed": False,
            "line_geometry_must_stay_inside_stage1_voxel_cells": True,
            "pole_attachment_requires_direct_stage1_contact": True,
            "open_line_endpoints_preserved": True,
        },
        "verification": verification,
        "tensorrt": {
            "benchmark": timing, "plan": plan,
            "deployment_note": "TensorRT validates/benchmarks ONNX on Nebius only. Sentis imports ONNX, not TensorRT engines.",
        },
        "sources": {
            "checkpoint_sha256": sha256(checkpoint), "stage2_profile_sha256": sha256(profile_path),
            "exporter": Path(__file__).name,
        },
        "host": {"python": platform.python_version(), "torch": torch.__version__},
    }
    sidecar_path = output / "v10_stage12_sidecar.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    (output / "UNITY_READINESS.txt").write_text(
        "V10 UNITY SENTIS EXPORT\n"
        f"FP32_READY=true\nFP16_READY={str(fp16_approved).lower()}\n"
        "STAGE2_REFERENCE=voxel-supported-opt1-fix2\n"
        "TENSORRT_USE=NEBIUS_BENCHMARK_ONLY_NOT_SENTIS_INPUT\n"
        "UNITY_VALIDATION_REQUIRED=true\n"
    )
    print("V10_STAGE1_ONNX_EXPORT_OK")
    print(json.dumps({"output": str(output), "fp16_deployable": fp16_approved,
                      "verification": verification}, indent=2))


if __name__ == "__main__":
    main()
