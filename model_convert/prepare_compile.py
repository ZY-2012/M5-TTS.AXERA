#!/usr/bin/env python3
"""Pack calibration sets and build Pulsar2 8.0 (Magnetar) configs for M5-TTS.

    python compile/prepare_compile.py                 # write configs only
    python compile/prepare_compile.py --models depth_prefill --run
    python compile/prepare_compile.py --run           # build every graph
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import tarfile

import onnx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX = os.path.join(ROOT, "export", "onnx")
CALIB = os.path.join(ROOT, "calib", "data")
COMPILE = os.path.join(ROOT, "compile")
PULSAR2 = os.environ.get("PULSAR2", "/root/.cache/magnetar/pulsar2/8.0/bin/pulsar2")

DTYPE = "U16"
WEIGHT_DTYPE = "S8"
SMOOTH_QUANT = True
PRECISION_ANALYSIS = True

SPECS = {
    "lm_prefill": dict(onnx="lm_prefill.onnx", npu="NPU1", out="lm_prefill.axmodel"),
    "lm_decode": dict(onnx="lm_decode.onnx", npu="NPU1", out="lm_decode.axmodel"),
    "depth_prefill": dict(onnx="depth_prefill.onnx", npu="NPU1", out="depth_prefill.axmodel"),
    "depth_step": dict(onnx="depth_step.onnx", npu="NPU1", out="depth_step.axmodel"),
    "codec_encoder": dict(onnx="codec_encoder.onnx", npu="NPU3", out="codec_encoder.axmodel"),
    "codec_decoder": dict(onnx="codec_decoder.onnx", npu="NPU3", out="codec_decoder.axmodel"),
}


def io_spec(path):
    model = onnx.load(path, load_external_data=False)
    names, shapes = [], []
    for value in model.graph.input:
        dims = [d.dim_value for d in value.type.tensor_type.shape.dim]
        names.append(value.name)
        shapes.append("x".join(str(d) for d in dims))
    return names, shapes


def pack(name, inputs):
    cdir = os.path.join(COMPILE, "calib_packed", name)
    os.makedirs(cdir, exist_ok=True)
    info = {}
    for tensor in inputs:
        files = sorted(glob.glob(os.path.join(CALIB, name, tensor, "*.npy")))
        assert files, f"missing calibration for {name}/{tensor}"
        tar = os.path.join(cdir, f"{tensor}.tar.gz")
        if not os.path.exists(tar) or len(files) != info.get("_n"):
            with tarfile.open(tar, "w:gz") as tf:
                for f in files:
                    tf.add(f, arcname=os.path.basename(f))
        info[tensor] = (tar, len(files))
    n = next(iter(info.values()))[1]
    assert all(v[1] == n for v in info.values()), (name, {k: v[1] for k, v in info.items()})
    return info, n


def make_config(name, spec):
    onnx_path = os.path.join(ONNX, spec["onnx"])
    names, shapes = io_spec(onnx_path)
    info, n = pack(name, names)
    cfg = {
        "input": onnx_path,
        "output_dir": COMPILE,
        "output_name": spec["out"],
        "work_dir": os.path.join(COMPILE, "work", name),
        "model_type": "ONNX",
        "target_hardware": "AX650",
        "npu_mode": os.environ.get("M5_NPU_MODE", spec["npu"]),
        "input_shapes": ";".join(f"{k}:{v}" for k, v in zip(names, shapes)),
        "onnx_opt": {"disable_onnx_optimization": False,
                     "enable_onnxsim": name == "codec_encoder",
                     "model_check": True},
        "quant": {
            "calibration_method": os.environ.get("M5_CALIB_METHOD", "MinMax"),
            "input_configs": [
                {"tensor_name": k, "calibration_dataset": info[k][0],
                 "calibration_format": "Numpy", "calibration_size": info[k][1],
                 "calibration_mean": [], "calibration_std": []}
                for k in names],
            "layer_configs": [
                {"start_tensor_names": ["DEFAULT"], "end_tensor_names": ["DEFAULT"],
                 "data_type": DTYPE, "weight_data_type": WEIGHT_DTYPE,
                 "output_data_type": DTYPE}
            ],
            "highest_mix_precision": False,
            "enable_smooth_quant": SMOOTH_QUANT,
            "precision_analysis": PRECISION_ANALYSIS,
            "precision_analysis_method": "EndToEnd",
            **({"transformer_opt_level": 1} if name.startswith(("lm_", "depth_")) else {}),
        },
        "compiler": {"check": 0},
    }
    path = os.path.join(COMPILE, f"pulsar2_config_{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"{name}: {n} calib samples, {len(names)} inputs -> {path}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    names = a.models.split(",") if a.models else list(SPECS)
    os.makedirs(os.path.join(COMPILE, "logs"), exist_ok=True)
    for name in names:
        cfg = make_config(name, SPECS[name])
        if a.run:
            log = os.path.join(COMPILE, "logs", f"{name}.log")
            print(f"== build {name} (log {log})", flush=True)
            with open(log, "w") as f:
                r = subprocess.run([PULSAR2, "build", "--config", cfg], stdout=f,
                                   stderr=subprocess.STDOUT)
            print("   rc", r.returncode, flush=True)
            os.system(f"tail -n 4 {log}")


if __name__ == "__main__":
    main()
