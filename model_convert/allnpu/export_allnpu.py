#!/usr/bin/env python3
"""Export the four all-NPU graph variants and validate them against torch.

    python export/export_allnpu.py            # export + torch/ONNX cosine
    python export/export_allnpu.py --check    # checks only

Reuses the checkpoint loaders and the unchanged graphs from
``../../M5-TTS-distill/export``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

HERE = Path(__file__).resolve().parents[1]
DISTILL = HERE.parent
sys.path.insert(0, str(DISTILL))
sys.path.insert(0, str(HERE / "export"))

import common as C
from common import CAP, CODEC_FRAMES, DEPTH_SLOTS, ONNX_DIR as DISTILL_ONNX, SEQ
from m5_graphs import CodecDecoder, DepthPrefill, DepthStep, Prefill, Decode

from allnpu_graphs import (AudioEmbedFrame, CodecDecoderRVQ, DepthPrefillHead,
                           DepthStepHeads, codec_indices, flat_indices)

OUT = HERE / "export" / "onnx"
AUDIO_VOCAB = 2051
CODEC_VOCAB = 2048
CHECKS = {}


def cosine(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 8
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def export(module, inputs, path, names, outputs):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(module.eval(), inputs, str(path), input_names=names, output_names=outputs,
                      opset_version=17, dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(str(path))
    print(f"exported {path.name}  {path.stat().st_size/1e6:.1f} MB", flush=True)


def rename_codec_states(path, count):
    """Exporter names the lifted conv states ``conv_i.1``; make them ``state_i``."""
    import onnx as _onnx
    m = _onnx.load(str(path))
    rename = {f"conv_{i}.1": f"state_{i}" for i in range(count)}
    for inp in m.graph.input:
        if inp.name in rename:
            inp.name = rename[inp.name]
    for node in m.graph.node:
        node.input[:] = [rename.get(x, x) for x in node.input]
    _onnx.save(m, str(path))


def check(name, sess, feed, expected):
    got = sess.run(None, {k: np.ascontiguousarray(v) for k, v in feed.items()})
    row = {}
    for index, (ref, val) in enumerate(zip(expected, got)):
        ref = ref.detach().cpu().numpy() if isinstance(ref, torch.Tensor) else np.asarray(ref)
        if np.issubdtype(ref.dtype, np.integer):
            row[f"out{index}"] = "int-exact" if np.array_equal(ref, val) else "INT-MISMATCH"
            assert np.array_equal(ref, val), (name, index)
        else:
            row[f"out{index}"] = round(cosine(ref, val), 8)
            assert cosine(ref, val) > 0.9999, (name, index, cosine(ref, val))
    CHECKS[name] = row
    print(f"  {name:16s} {row}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    assets = C.ASSET_DIR
    audio_emb = np.load(assets / "audio_embeddings.npy")
    proj = np.load(assets / "projection.npy")
    head0 = np.load(assets / "head0.npy")
    audio_head = np.load(assets / "audio_head.npy")
    z = np.load(assets / "codec_rvq.npz")

    graphs = {
        "audio_embed": AudioEmbedFrame(audio_emb, proj),
    }
    if args.check:
        # graphs only needed for the checks; loading the checkpoints is slow
        model = C.load_lm()[0]
        codec = C.load_codec()
        graphs["depth_prefill"] = DepthPrefillHead(DepthPrefill(model.decoder), audio_head[0])
        graphs["depth_step"] = DepthStepHeads(DepthStep(model.decoder), audio_head)
        graphs["codec_decoder"] = CodecDecoderRVQ(
            codec.decoder, z["book_sem"][0], z["book_rest"],
            z["proj_sem_w"], z["proj_sem_b"], z["proj_rest_w"], z["proj_rest_b"])
    else:
        model = C.load_lm()[0]
        codec = C.load_codec()
        graphs["depth_prefill"] = DepthPrefillHead(DepthPrefill(model.decoder), audio_head[0])
        graphs["depth_step"] = DepthStepHeads(DepthStep(model.decoder), audio_head)
        graphs["codec_decoder"] = CodecDecoderRVQ(
            codec.decoder, z["book_sem"][0], z["book_rest"],
            z["proj_sem_w"], z["proj_sem_b"], z["proj_rest_w"], z["proj_rest_b"])

        # ---------------------------------------------------------- audio_embed
        i0 = np.array([[3 * AUDIO_VOCAB + 7, 2051 + 5, 2 * AUDIO_VOCAB + 1, 9,
                        4 * AUDIO_VOCAB + 100, 5 * AUDIO_VOCAB, 6 * AUDIO_VOCAB + 2000,
                        7 * AUDIO_VOCAB + 42]], np.int32)
        i1 = np.array([[11, AUDIO_VOCAB + 3, 2 * AUDIO_VOCAB + 77,
                        3 * AUDIO_VOCAB + 9, 4 * AUDIO_VOCAB + 500, 5 * AUDIO_VOCAB + 1,
                        6 * AUDIO_VOCAB + 2040, 7 * AUDIO_VOCAB + 123]], np.int32)
        g = graphs["audio_embed"]
        expected = g(torch.from_numpy(i0), torch.from_numpy(i1))
        export(g, (torch.from_numpy(i0), torch.from_numpy(i1)), OUT / "audio_embed.onnx",
               ["i0", "i1"], ["emb", "local"])
        check("audio_embed", session(OUT / "audio_embed.onnx"), {"i0": i0, "i1": i1}, expected)

        # ------------------------------------------------------- depth_prefill
        local = torch.randn(1, 2, 512)
        g = graphs["depth_prefill"]
        expected = g(local)
        export(g, (local,), OUT / "depth_prefill.onnx", ["local"], ["h", "logits", "kv"])
        check("depth_prefill", session(OUT / "depth_prefill.onnx"), {"local": local.numpy()}, expected)

        # ---------------------------------------------------------- depth_step
        slots = DEPTH_SLOTS
        past = torch.zeros(len(model.decoder.layers), 2, 1,
                           model.decoder.config.num_key_value_heads, slots, 64)
        past[:, :, :, :, :2] = torch.randn(len(model.decoder.layers), 2, 1,
                                           model.decoder.config.num_key_value_heads, 2, 64)
        cos_t, sin_t = C.rope_cos_sin(model.decoder.position_embedding.inv_freq.detach().numpy(),
                                      float(model.decoder.position_embedding.attention_scaling), [2])
        cos_t = torch.from_numpy(cos_t).view(1, 1, 1, -1)
        sin_t = torch.from_numpy(sin_t).view(1, 1, 1, -1)
        mask = np.full((1, 1, 1, slots + 1), -30.0, np.float32)
        mask[..., :2] = 0.0
        mask[..., -1] = 0.0
        mask_t = torch.from_numpy(mask)
        dlocal = torch.randn(1, 1, 512)
        onehot = torch.zeros(14)
        onehot[6] = 1.0
        g = graphs["depth_step"]
        expected = g(dlocal, onehot, cos_t, sin_t, mask_t, past)
        export(g, (dlocal, onehot, cos_t, sin_t, mask_t, past), OUT / "depth_step.onnx",
               ["local", "onehot", "cos", "sin", "mask", "past"], ["h", "logits", "kv"])
        check("depth_step", session(OUT / "depth_step.onnx"),
              {"local": dlocal.numpy(), "onehot": onehot.numpy(), "cos": cos_t.numpy(),
               "sin": sin_t.numpy(), "mask": mask_t.numpy(), "past": past.numpy()}, expected)

        # ------------------------------------------------------- codec_decoder
        dec = graphs["codec_decoder"]
        frames = CODEC_FRAMES
        codes = np.random.default_rng(3).integers(0, CODEC_VOCAB, size=(frames, 16)).astype(np.int32)
        j0, j1 = codec_indices(codes)
        past = torch.zeros(dec.layers, 2, 1, dec.kv_heads, dec.window - 1, dec.dim)
        states = [torch.zeros(spec["shape"]) for spec in dec.specs]
        cos_c, sin_c = C.rope_cos_sin(codec.decoder.pre_transformer.rotary_emb.inv_freq.detach().numpy(),
                                      float(codec.decoder.pre_transformer.rotary_emb.attention_scaling),
                                      np.arange(frames))
        cos_t = torch.from_numpy(cos_c).view(1, 1, -1, dec.dim)
        sin_t = torch.from_numpy(sin_c).view(1, 1, -1, dec.dim)
        key_abs = np.arange(dec.window - 1 + frames) - (dec.window - 1)
        pos_abs = np.arange(frames)
        allowed = ((key_abs[None, :] <= pos_abs[:, None]) & (key_abs[None, :] >= 0))
        mask_t = torch.from_numpy(np.where(allowed, 0.0, -30.0).astype(np.float32)[None, None])
        g = graphs["codec_decoder"]
        inputs = (torch.from_numpy(j0)[None], torch.from_numpy(j1)[None], past, cos_t, sin_t,
                  mask_t, *states)
        expected = g(*inputs)
        names = ["i0", "i1", "past", "cos", "sin", "mask"] + [f"state_{i}" for i in range(len(states))]
        outs = ["wav", "present"] + [f"conv_{i}" for i in range(len(states))]
        export(g, inputs, OUT / "codec_decoder.onnx", names, outs)
        rename_codec_states(OUT / "codec_decoder.onnx", len(states))
        feed = {n: v.detach().cpu().numpy() for n, v in zip(names, inputs)}
        check("codec_decoder", session(OUT / "codec_decoder.onnx"), feed, expected)

    # ------------------------------------------------------------------ checks
    if args.check:
        s = session(OUT / "audio_embed.onnx")
        i0 = np.array([[0, 2051, 2 * AUDIO_VOCAB, 3 * AUDIO_VOCAB, 4 * AUDIO_VOCAB,
                        5 * AUDIO_VOCAB, 6 * AUDIO_VOCAB, 7 * AUDIO_VOCAB]], np.int32)
        i1 = np.array([[0, AUDIO_VOCAB, 2 * AUDIO_VOCAB, 3 * AUDIO_VOCAB,
                        4 * AUDIO_VOCAB, 5 * AUDIO_VOCAB, 6 * AUDIO_VOCAB, 7 * AUDIO_VOCAB]], np.int32)
        check("audio_embed", s, {"i0": i0, "i1": i1}, graphs["audio_embed"](torch.from_numpy(i0), torch.from_numpy(i1)))

        local = torch.randn(1, 2, 512)
        check("depth_prefill", session(OUT / "depth_prefill.onnx"), {"local": local.numpy()},
              graphs["depth_prefill"](local))

        slots = DEPTH_SLOTS
        past = torch.zeros(len(model.decoder.layers), 2, 1,
                           model.decoder.config.num_key_value_heads, slots, 64)
        cos_t, sin_t = C.rope_cos_sin(model.decoder.position_embedding.inv_freq.detach().numpy(),
                                      float(model.decoder.position_embedding.attention_scaling), [2])
        cos_t = torch.from_numpy(cos_t).view(1, 1, 1, -1)
        sin_t = torch.from_numpy(sin_t).view(1, 1, 1, -1)
        mask = np.full((1, 1, 1, slots + 1), -30.0, np.float32)
        mask[..., :2] = 0.0
        mask[..., -1] = 0.0
        mask_t = torch.from_numpy(mask)
        dlocal = torch.randn(1, 1, 512)
        onehot = torch.zeros(14)
        onehot[2] = 1.0
        check("depth_step", session(OUT / "depth_step.onnx"),
              {"local": dlocal.numpy(), "onehot": onehot.numpy(), "cos": cos_t.numpy(),
               "sin": sin_t.numpy(), "mask": mask_t.numpy(), "past": past.numpy()},
              graphs["depth_step"](dlocal, onehot, cos_t, sin_t, mask_t, past))

        dec = graphs["codec_decoder"]
        frames = CODEC_FRAMES
        codes = np.random.default_rng(5).integers(0, CODEC_VOCAB, size=(frames, 16)).astype(np.int32)
        j0, j1 = codec_indices(codes)
        past = torch.zeros(dec.layers, 2, 1, dec.kv_heads, dec.window - 1, dec.dim)
        states = [torch.zeros(spec["shape"]) for spec in dec.specs]
        cos_c, sin_c = C.rope_cos_sin(codec.decoder.pre_transformer.rotary_emb.inv_freq.detach().numpy(),
                                      float(codec.decoder.pre_transformer.rotary_emb.attention_scaling),
                                      np.arange(frames))
        cos_t = torch.from_numpy(cos_c).view(1, 1, -1, dec.dim)
        sin_t = torch.from_numpy(sin_c).view(1, 1, -1, dec.dim)
        key_abs = np.arange(dec.window - 1 + frames) - (dec.window - 1)
        pos_abs = np.arange(frames)
        allowed = ((key_abs[None, :] <= pos_abs[:, None]) & (key_abs[None, :] >= 0))
        mask_t = torch.from_numpy(np.where(allowed, 0.0, -30.0).astype(np.float32)[None, None])
        inputs = (torch.from_numpy(j0)[None], torch.from_numpy(j1)[None], past, cos_t, sin_t,
                  mask_t, *states)
        names = ["i0", "i1", "past", "cos", "sin", "mask"] + [f"state_{i}" for i in range(len(states))]
        feed = {n: v.detach().cpu().numpy() for n, v in zip(names, inputs)}
        check("codec_decoder", session(OUT / "codec_decoder.onnx"), feed, graphs["codec_decoder"](*inputs))

    meta = {
        "audio_vocab": AUDIO_VOCAB,
        "codec_vocab": CODEC_VOCAB,
        "audio_embed": {"inputs": ["i0", "i1"], "outputs": ["emb", "local"],
                        "index": "level*2051+code, levels 0-7 in i0, 8-15 in i1"},
        "codec_decoder": {"inputs": ["i0", "i1", "past", "cos", "sin", "mask"] +
                          [f"conv_{i}" for i in range(len(dec.specs))],
                          "index": "i0=code0, i1=(level-1)*2048+code, T frames; states are state_0..state_20"},
        "checks": CHECKS,
    }
    (HERE / "export" / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("wrote export/meta.json")


if __name__ == "__main__":
    main()
