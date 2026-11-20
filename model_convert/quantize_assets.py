#!/usr/bin/env python3
"""Store the host-side tables as int8 + per-group scales (4x smaller package).

Each table is saved in its *runtime layout*; every group of `inner` consecutive
floats (inner = 256 for codebooks, 512 for embeddings/heads) shares one scale.
The runtimes dequantize to fp32 at load, so the math is unchanged; only the
download/package size shrinks.

    python export/quantize_assets.py --src export/assets --dst sdk/assets_q8
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def qsave(arr, dst, name, inner):
    a = np.asarray(arr, np.float32)
    assert a.shape[-1] == inner, (name, a.shape, inner)
    m = a.reshape(-1, inner)
    scale = np.maximum(np.abs(m).max(axis=1) / 127.0, 1e-12).astype(np.float32)
    q = np.clip(np.round(m / scale[:, None]), -127, 127).astype(np.int8).reshape(a.shape)
    np.save(dst / f"{name}_i8.npy", q)
    np.save(dst / f"{name}_scale.npy", scale)
    rel = float(np.sqrt(((q.reshape(-1, inner) * scale[:, None] - m) ** 2).mean()) /
                (np.sqrt((m ** 2).mean()) + 1e-12))
    old = (dst.parent / f"{name}.npy")
    return rel, q.nbytes + scale.nbytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="export/assets")
    ap.add_argument("--dst", default="sdk/assets_q8")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)

    plans = {
        "text_embeddings": 512,
        "audio_embeddings": 512,
        "head0": 512,
        "projected_audio": 512,
        "dec_sem_book": 256,
        "dec_rest_book": 256,
        "enc_sem_book": 256,
        "enc_ac_book": 256,
    }
    total = 0
    for name, inner in plans.items():
        arr = np.load(src / f"{name}.npy")
        rel, nbytes = qsave(arr, dst, name, inner)
        print(f"{name:20s} {arr.shape} int8  rel-err {rel*100:5.2f}%  {arr.nbytes/1e6:6.1f} -> {nbytes/1e6:5.1f} MB")
        total += nbytes

    # audio_head: transpose to [level][vocab][dim] (the runtime layout) before quantizing
    ah = np.load(src / "audio_head.npy")
    rel, nbytes = qsave(np.ascontiguousarray(np.transpose(ah, (0, 2, 1))), dst, "audio_head", 512)
    print(f"{'audio_head':20s} {ah.shape}->[15,2051,512] int8 rel-err {rel*100:5.2f}%  "
          f"{ah.nbytes/1e6:6.1f} -> {nbytes/1e6:5.1f} MB")
    total += nbytes
    print(f"tables total: {total/1e6:.0f} MB")

    for name in ("projection.npy", "backbone_inv_freq.npy", "decoder_inv_freq.npy", "codec_inv_freq.npy",
                 "codebook_prior.npy", "mean_code.npy", "config.json", "codec_meta.json",
                 "tokenizer.json", "tokenizer.bin", "dec_proj_sem_w.npy", "dec_proj_sem_b.npy",
                 "dec_proj_rest_w.npy", "dec_proj_rest_b.npy", "enc_sem_w.npy", "enc_ac_w.npy"):
        p = src / name
        if p.is_file():
            shutil.copy2(p, dst / name)
    print("wrote", dst, f"({sum(p.stat().st_size for p in dst.iterdir())/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
