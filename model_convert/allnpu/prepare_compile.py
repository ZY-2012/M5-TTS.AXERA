#!/usr/bin/env python3
"""Build Pulsar2 configs for the four all-NPU graph variants.

    python compile/prepare_compile.py            # configs + calibration
    python compile/prepare_compile.py --run      # build every graph
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import onnx

HERE = Path(__file__).resolve().parents[1]
DISTILL = HERE.parent / "M5-TTS-distill"
ONNX_DIR = HERE / "export" / "onnx"
CALIB = HERE / "calib" / "data"
COMPILE = HERE / "compile"
PULSAR2 = os.environ.get("PULSAR2", "/root/.cache/magnetar/pulsar2/8.0/bin/pulsar2")

SPECS = {
    "audio_embed": dict(npu="NPU1"),
    "depth_prefill": dict(npu="NPU1"),
    "depth_step": dict(npu="NPU1"),
    "codec_decoder": dict(npu="NPU3"),
}


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        return
    dst.symlink_to(src)


def prepare_calib():
    # reuse the unchanged tensors from the distill calibration set
    for name in ("local", "cos", "sin", "mask", "past"):
        link(DISTILL / "calib/data/depth_step" / name, CALIB / "depth_step" / name)
    link(DISTILL / "calib/data/depth_prefill/local", CALIB / "depth_prefill/local")
    for name in ("cos", "sin", "mask", "past"):
        link(DISTILL / "calib/data/codec_decoder" / name, CALIB / "codec_decoder" / name)
    for i in range(21):
        link(DISTILL / f"calib/data/codec_decoder/conv_{i}", CALIB / f"codec_decoder/state_{i}")

    rng = np.random.default_rng(0)
    for sub in ("audio_embed/i0", "audio_embed/i1", "depth_step/onehot",
                "codec_decoder/i0", "codec_decoder/i1"):
        (CALIB / sub).mkdir(parents=True, exist_ok=True)

    # audio_embed: 16 codebooks x 2051 entries; 32 samples, 8 codes per table
    for i in range(32):
        codes = rng.integers(0, 2051, size=(1, 16)).astype(np.int32)
        i0 = codes[:, :8] + (np.arange(8, dtype=np.int32) * 2051)[None]
        i1 = codes[:, 8:] + (np.arange(8, dtype=np.int32) * 2051)[None]
        np.save(CALIB / f"audio_embed/i0/{i:05d}.npy", i0)
        np.save(CALIB / f"audio_embed/i1/{i:05d}.npy", i1)

    # depth_step: 14-way one-hot, 64 samples
    for i in range(64):
        v = np.zeros(14, np.float32)
        v[i % 14] = 1.0
        np.save(CALIB / f"depth_step/onehot/{i:05d}.npy", v)

    # codec_decoder: 8 frames, 16 codebooks
    for i in range(32):
        codes = rng.integers(0, 2048, size=(8, 16)).astype(np.int32)
        np.save(CALIB / f"codec_decoder/i0/{i:05d}.npy", codes[:, 0][None])
        off = (np.arange(15, dtype=np.int32) * 2048)[None]
        np.save(CALIB / f"codec_decoder/i1/{i:05d}.npy", (codes[:, 1:] + off).reshape(1, -1))


def io_spec(path):
    model = onnx.load(str(path), load_external_data=False)
    names, shapes = [], []
    for value in model.graph.input:
        dims = [d.dim_value for d in value.type.tensor_type.shape.dim]
        names.append(value.name)
        shapes.append("x".join(str(d) for d in dims))
    return names, shapes


def pack(name, inputs):
    cdir = COMPILE / "calib_packed" / name
    cdir.mkdir(parents=True, exist_ok=True)
    info = {}
    for tensor in inputs:
        files = sorted(glob.glob(str(CALIB / name / tensor / "*.npy")))
        assert files, f"missing calibration for {name}/{tensor}"
        tar = cdir / f"{tensor}.tar.gz"
        if not tar.exists() or len(files) != info.get("_n"):
            with tarfile.open(tar, "w:gz") as tf:
                for f in files:
                    tf.add(f, arcname=os.path.basename(f))
        info[tensor] = (str(tar), len(files))
    n = next(iter(info.values()))[1]
    assert all(v[1] == n for v in info.values()), (name, {k: v[1] for k, v in info.items()})
    return info, n


def make_config(name):
    onnx_path = ONNX_DIR / f"{name}.onnx"
    names, shapes = io_spec(onnx_path)
    info, n = pack(name, names)
    cfg = {
        "input": str(onnx_path),
        "output_dir": str(COMPILE),
        "output_name": f"{name}.axmodel",
        "work_dir": str(COMPILE / "work" / name),
        "model_type": "ONNX",
        "target_hardware": "AX650",
        "npu_mode": os.environ.get("M5_NPU_MODE", SPECS[name]["npu"]),
        "input_shapes": ";".join(f"{k}:{v}" for k, v in zip(names, shapes)),
        "onnx_opt": {"disable_onnx_optimization": False,
                     "enable_onnxsim": False,
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
                 "data_type": "U16", "weight_data_type": os.environ.get("M5_WEIGHT_DTYPE", "S8"),
                 "output_data_type": "U16"}
            ],
            "highest_mix_precision": False,
            "enable_smooth_quant": True,
            "precision_analysis": True,
            "precision_analysis_method": "EndToEnd",
            **({"transformer_opt_level": 1} if name in ("depth_prefill", "depth_step") else {}),
        },
        "compiler": {"check": 0},
    }
    path = COMPILE / f"pulsar2_config_{name}.json"
    path.write_text(json.dumps(cfg, indent=2))
    print(f"{name}: {n} calib samples, {len(names)} inputs -> {path.name}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    prepare_calib()
    names = a.models.split(",") if a.models else list(SPECS)
    (COMPILE / "logs").mkdir(parents=True, exist_ok=True)
    for name in names:
        cfg = make_config(name)
        if a.run:
            log = COMPILE / "logs" / f"{name}.log"
            print(f"building {name} -> {log}", flush=True)
            with log.open("w") as fh:
                code = subprocess.call([PULSAR2, "build", "--config", str(cfg)],
                                       stdout=fh, stderr=subprocess.STDOUT)
            print(f"{name}: {'ok' if code == 0 else f'FAILED {code}'}  "
                  f"{(COMPILE / (name + '.axmodel')).stat().st_size/1e6 if code == 0 else 0:.1f} MB",
                  flush=True)


if __name__ == "__main__":
    main()
