#!/usr/bin/env python3
"""AX650 per-graph accuracy check: axmodel vs the FP32 ONNX reference I/O.

    python3 board_check.py [--models models] [--eval eval_io]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m5_ax.tts import AxEngine  # noqa: E402

GRAPHS = ["lm_prefill", "lm_decode", "depth_prefill", "depth_step",
          "codec_encoder", "codec_decoder"]


def metrics(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    snr = float(10 * np.log10(np.sum(a * a) / (np.sum((a - b) ** 2) + 1e-12)))
    return cos, snr


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(here, "models"))
    ap.add_argument("--eval", default=os.path.join(here, "eval_io"))
    a = ap.parse_args()
    rows = []
    for name in GRAPHS:
        path = os.path.join(a.models, name + ".axmodel")
        folder = os.path.join(a.eval, name)
        if not os.path.exists(path) or not os.path.isdir(folder):
            print(f"== {name}: missing, skip")
            continue
        engine = AxEngine(path)
        ins = {f[:-4][3:]: np.load(os.path.join(folder, f)) for f in sorted(os.listdir(folder))
               if f.startswith("in_")}
        outs = {f[:-4][4:]: np.load(os.path.join(folder, f)) for f in sorted(os.listdir(folder))
                if f.startswith("out_")}
        feeds = []
        for i in range(next(iter(ins.values())).shape[0]):
            feeds.append({k: v[i] for k, v in ins.items()})
        got_all = [engine.run(feed) for feed in feeds]
        names = engine.output_names
        print(f"== {name}")
        for i in range(len(feeds)):
            for out_name, value in zip(names, got_all[i]):
                if out_name not in outs:
                    continue
                ref = outs[out_name][i]
                got = np.asarray(value).reshape(ref.shape)
                if np.issubdtype(ref.dtype, np.integer):
                    same = float(np.mean(np.round(got) == ref))
                    rows.append((name, out_name, i, same, 99.0))
                    print(f"  s{i} {out_name:12s} int-match {same*100:.3f}%")
                else:
                    cos, snr = metrics(ref, got)
                    rows.append((name, out_name, i, cos, snr))
                    print(f"  s{i} {out_name:12s} cosine {cos:.6f}  SNR {snr:6.2f} dB")
    bad = [r for r in rows if (r[3] < 0.99 and r[3] <= 1.0) or (r[3] <= 1.0 and r[4] < 20 and r[4] != 99.0)]
    print(f"\n{len(rows)} tensors checked, {len(bad)} below 0.99/20 dB")
    for r in bad:
        print("  LOW", r)


if __name__ == "__main__":
    main()
