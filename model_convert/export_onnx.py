#!/usr/bin/env python3
"""Export the static-shape M5-TTS ONNX bundle (six graphs) and validate it.

    python export/export_onnx.py            # export + torch/ONNX cosine check
    python export/export_onnx.py --check    # re-run the checks only
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "export"))
import common as C
from common import CAP, CODEC_FRAMES, CODEC_SAMPLES, DEPTH_SLOTS, SEQ, ONNX_DIR
from m5_graphs import (CodecDecoder, CodecEncoder, Decode, DepthPrefill, DepthStep,
                       EncoderTransformer, Prefill, StaticMimiConv1d, causal_extra_padding, export_graph)

CHECKS = {}


def hf_prefill(decoder, hidden):
    """HF llama stack (correct plural-cache kwarg) — the numerical ground truth."""
    from transformers.cache_utils import DynamicCache
    seq = hidden.shape[1]
    cache = DynamicCache()
    positions = torch.arange(seq)[None]
    mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))[None, None]
    emb = decoder.position_embedding(hidden, positions)
    x = hidden
    for layer in decoder.layers:
        x = layer(hidden_states=x, attention_mask=mask, position_embeddings=emb,
                  use_cache=True, past_key_values=cache, cache_position=torch.arange(seq))
    return decoder.norm(x), cache


def hf_step(decoder, hidden, cache, position):
    positions = torch.tensor([[position]])
    mask = torch.ones(1, 1, 1, position + 1, dtype=torch.bool)
    emb = decoder.position_embedding(hidden, positions)
    x = hidden
    for layer in decoder.layers:
        x = layer(hidden_states=x, attention_mask=mask, position_embeddings=emb,
                  use_cache=True, past_key_values=cache, cache_position=torch.tensor([position]))
    return decoder.norm(x)


def cosine(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 8
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def check(name, sess, inputs, expected):
    feed = {spec.name: np.ascontiguousarray(value.detach().cpu().numpy())
            for spec, value in zip(sess.get_inputs(), inputs)}
    got = sess.run(None, feed)
    row = {}
    for index, (ref, val) in enumerate(zip(expected, got)):
        ref = ref.detach().cpu().numpy() if isinstance(ref, torch.Tensor) else ref
        if np.issubdtype(ref.dtype, np.integer):
            same = bool(np.array_equal(ref, val))
            row[f"out{index}"] = "int-exact" if same else "INT-MISMATCH"
            assert same, (name, index)
        else:
            row[f"out{index}"] = round(cosine(ref, val), 8)
            assert cosine(ref, val) > 0.999, (name, index, cosine(ref, val))
    CHECKS[name] = row
    print(f"  {name:14s} {row}", flush=True)


def run_checks(model, codec, graphs, with_codec=True):
    torch.set_grad_enabled(False)
    # ---------------- backbone prefill vs HF LlamaModel
    seq = SEQ
    torch.manual_seed(0)
    hidden = torch.randn(1, seq, 512)
    pos = torch.arange(seq)[None]
    prefill = graphs["lm_prefill"]
    expected_h, expected_kv = prefill(hidden, prefill.cos, prefill.sin)
    from transformers.cache_utils import DynamicCache
    ref, dc = hf_prefill(model.backbone, hidden)
    cos = cosine(expected_h, ref)
    print(f"  prefill hidden cosine vs HF: {cos:.8f}")
    assert cos > 0.9999
    legacy = dc.to_legacy_cache()
    ref_kv = torch.stack([torch.stack((k[0], v[0])) for k, v in legacy])
    cos_kv = cosine(expected_kv, ref_kv)
    print(f"  prefill KV     cosine vs HF: {cos_kv:.8f}")
    assert cos_kv > 0.9999
    s = session(ONNX_DIR / "lm_prefill.onnx")
    check("lm_prefill", s, [hidden, prefill.cos, prefill.sin], [expected_h, expected_kv])

    # ---------------- backbone decode vs HF with a cache
    cap = CAP
    pos0 = 40
    torch.manual_seed(1)
    dh = torch.randn(1, 1, 512)
    mask = np.full((1, 1, 1, cap + 1), -30.0, np.float32)
    mask[..., :pos0] = 0.0
    mask[..., -1] = 0.0
    mask_t = torch.from_numpy(mask)
    dc = DynamicCache()
    model.backbone(hidden_states=torch.randn(1, pos0, 512),
                   attention_mask=torch.tril(torch.ones(pos0, pos0, dtype=torch.bool))[None, None],
                   position_ids=torch.arange(pos0)[None], use_cache=True, past_key_values=dc)
    cos_t, sin_t = C.rope_cos_sin(model.backbone.position_embedding.inv_freq.detach().numpy(),
                                   float(model.backbone.position_embedding.attention_scaling), [pos0])
    cos_t, sin_t = torch.from_numpy(cos_t).view(1, 1, 1, -1), torch.from_numpy(sin_t).view(1, 1, 1, -1)
    past = torch.zeros(len(model.backbone.layers), 2, 1, model.backbone.config.num_key_value_heads, cap, 64)
    for index, (k, v) in enumerate(dc.to_legacy_cache()):
        past[index, 0, :, :, :pos0] = k[0]
        past[index, 1, :, :, :pos0] = v[0]
    decode = graphs["lm_decode"]
    exp_h, exp_kv = decode(dh, cos_t, sin_t, mask_t, past)
    ref_h = hf_step(model.backbone, dh, dc, pos0)
    print(f"  decode hidden cosine vs HF: {cosine(exp_h, ref_h):.8f}")
    s = session(ONNX_DIR / "lm_decode.onnx")
    check("lm_decode", s, [dh, cos_t, sin_t, mask_t, past], [exp_h, exp_kv])

    # ---------------- depth prefill / step vs HF decoder
    local = torch.randn(1, 2, 512)
    dp = graphs["depth_prefill"]
    exp_h, exp_kv = dp(local)
    ref, dc = hf_prefill(model.decoder, local)
    print(f"  depth prefill cosine vs HF: {cosine(exp_h, ref[:, -1]):.8f}")
    s = session(ONNX_DIR / "depth_prefill.onnx")
    check("depth_prefill", s, [local], [exp_h, exp_kv])

    slots = DEPTH_SLOTS
    past = torch.zeros(len(model.decoder.layers), 2, 1, model.decoder.config.num_key_value_heads, slots, 64)
    past[:, :, :, :, :2] = exp_kv
    step = graphs["depth_step"]
    pos = 2
    cos_t, sin_t = C.rope_cos_sin(model.decoder.position_embedding.inv_freq.detach().numpy(),
                                   float(model.decoder.position_embedding.attention_scaling), [pos])
    cos_t, sin_t = torch.from_numpy(cos_t).view(1, 1, 1, -1), torch.from_numpy(sin_t).view(1, 1, 1, -1)
    mask = np.full((1, 1, 1, slots + 1), -30.0, np.float32)
    mask[..., :pos] = 0.0
    mask[..., -1] = 0.0
    mask_t = torch.from_numpy(mask)
    dlocal = torch.randn(1, 1, 512)
    exp_h, exp_kv = step(dlocal, cos_t, sin_t, mask_t, past)
    ref = hf_step(model.decoder, dlocal, dc, pos)
    print(f"  depth step cosine vs HF:    {cosine(exp_h, ref[:, 0]):.8f}")
    s = session(ONNX_DIR / "depth_step.onnx")
    check("depth_step", s, [dlocal, cos_t, sin_t, mask_t, past], [exp_h, exp_kv])

    if not with_codec:
        return

    # ---------------- codec encoder vs torch
    torch.manual_seed(2)
    audio = torch.sin(torch.arange(CODEC_SAMPLES, dtype=torch.float32) * 0.01)[None] * 0.1
    enc = graphs["codec_enc"]
    exp_feat = enc(audio)
    s = session(ONNX_DIR / "codec_encoder.onnx")
    check("codec_encoder", s, [audio], [exp_feat])
    if hasattr(codec, "_ref_codes"):
        evq = C.EncoderVQNP(C.ASSET_DIR / "codec_enc_vq.npz")
        ref_audio = C.read_reference(C.ROOT.parents[1] / "M5-TTS" / "assets" / "aslp_default_voice.wav")
        ref_audio = np.pad(ref_audio[:CODEC_SAMPLES], (0, max(0, CODEC_SAMPLES - len(ref_audio))))
        feat = s.run(None, {"audio": ref_audio[None].astype(np.float32)})[0]
        got = evq.encode(feat[0])
        ref = codec._ref_codes.numpy().reshape(16, -1)
        same = float(np.mean(got == ref))
        print(f"  codec encoder (host RVQ) vs original model codes: {same*100:.3f}% exact")
        assert same > 0.999

    # ---------------- streaming decoder vs full decode
    dec = graphs["codec_dec"]
    rvq = C.RVQ(codec.decoder)
    frames = 12
    codes = np.random.default_rng(3).integers(0, 2048, size=(16, frames)).astype(np.int64)
    latent = torch.from_numpy(rvq.decode(codes))[None]
    ref_wav = full_codec_decode(codec, latent)
    offset = 0
    past = torch.zeros(dec.layers, 2, 1, dec.kv_heads, dec.window - 1, dec.dim)
    states = [torch.zeros(spec["shape"]) for spec in dec.specs]
    chunks = []
    sess = session(ONNX_DIR / "codec_decoder.onnx")
    for start in range(0, frames, CODEC_FRAMES):
        n = min(CODEC_FRAMES, frames - start)
        chunk = torch.zeros(1, latent.shape[1], CODEC_FRAMES)
        chunk[:, :, :n] = latent[:, :, start:start + n]
        positions = np.arange(offset, offset + CODEC_FRAMES)
        cos_c, sin_c = C.rope_cos_sin(codec.decoder.pre_transformer.rotary_emb.inv_freq.detach().numpy(),
                                      float(codec.decoder.pre_transformer.rotary_emb.attention_scaling),
                                      positions)
        cos_t = torch.from_numpy(cos_c).view(1, 1, -1, 64)
        sin_t = torch.from_numpy(sin_c).view(1, 1, -1, 64)
        key_abs = np.arange(dec.window - 1 + CODEC_FRAMES) + offset - (dec.window - 1)
        pos_abs = offset + np.arange(CODEC_FRAMES)
        allowed = ((key_abs[None, :] <= pos_abs[:, None]) & (key_abs[None, :] >= 0) &
                   (key_abs[None, :] >= pos_abs[:, None] - (dec.window - 1)))
        mask_t = torch.from_numpy(np.where(allowed, 0.0, -30.0).astype(np.float32)[None, None])
        feed = [chunk, past, cos_t, sin_t, mask_t, *states]
        expected = dec(chunk, past, cos_t, sin_t, mask_t, *states)
        names = ["latent", "past", "cos", "sin", "mask"] + [f"conv_{i}" for i in range(len(states))]
        out = sess.run(None, {n: np.ascontiguousarray(v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v)
                              for n, v in zip(names, feed)})
        assert cosine(expected[0], out[0]) > 0.9999, cosine(expected[0], out[0])
        chunks.append(torch.from_numpy(out[0]).reshape(-1)[:n * 1920])
        past = torch.from_numpy(out[1])
        states = [torch.from_numpy(x) for x in out[2:]]
        offset += CODEC_FRAMES
    streamed = torch.cat(chunks, dim=-1)
    cos = cosine(streamed, ref_wav[..., :streamed.shape[-1]])
    print(f"  codec_decoder stream vs full: {cos:.8f}")
    assert cos > 0.999


def full_codec_decode(codec, latent):
    """Reference (non-streaming) decoder path starting from a summed latent."""
    d = codec.decoder
    hidden = d.pre_conv(latent).transpose(1, 2)
    hidden = d.pre_transformer(inputs_embeds=hidden).last_hidden_state.permute(0, 2, 1)
    for blocks in d.upsample:
        for block in blocks:
            hidden = block(hidden)
    for block in d.decoder:
        hidden = block(hidden)
    return hidden.clamp(-1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--skip-codec", action="store_true")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    print("loading checkpoints ...", flush=True)
    model, tokenizer = C.load_lm()
    codec = C.load_codec()
    from types import MethodType
    from m5.codec_qwen12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2CausalConvNet
    for module in codec.decoder.modules():
        if type(module).__name__ == "Qwen3TTSTokenizerV2CausalConvNet":
            module._get_extra_padding_for_conv1d = MethodType(causal_extra_padding, module)
    for module in codec.encoder.modules():
        if type(module).__name__ == "Qwen3TTSTokenizerV2CausalConvNet":
            module._get_extra_padding_for_conv1d = MethodType(causal_extra_padding, module)
    codec.decoder.pre_transformer.config._attn_implementation = "eager"
    # reference codes from the untouched encoder, for an end-to-end identity check
    ref_audio = C.read_reference(C.ROOT.parents[1] / "M5-TTS" / "assets" / "aslp_default_voice.wav")
    ref_audio = np.pad(ref_audio[:CODEC_SAMPLES], (0, max(0, CODEC_SAMPLES - len(ref_audio))))
    with torch.inference_mode():
        x = codec.encoder.encoder(torch.from_numpy(ref_audio)[None, None], padding_cache=None)
        x = codec.encoder.encoder_transformer(x.transpose(1, 2))[0].transpose(1, 2)
        x = codec.encoder.downsample(x, padding_cache=None)
        ref_codes = codec.encoder.quantizer.encode(x, 16).detach()
    codec.encoder.encoder_transformer.config._attn_implementation = "eager"
    codec.encoder.encoder_transformer = EncoderTransformer(codec.encoder.encoder_transformer, 120)
    # replace every MimiConv1d with a static replicate-pad version (Pulsar2 has
    # no edge-mode Pad); capture each module's input length first
    from transformers.models.mimi.modeling_mimi import MimiConv1d
    codec.encoder.downsample = StaticMimiConv1d(codec.encoder.downsample, 120)
    lengths, hooks = {}, []

    def record_length(module, inp, out, store=lengths):
        store[id(module)] = int(inp[0].shape[-1])

    for module in codec.encoder.modules():
        if isinstance(module, MimiConv1d):
            hooks.append(module.register_forward_hook(record_length))
    with torch.inference_mode():
        codec.encoder.encoder(torch.from_numpy(ref_audio)[None, None], padding_cache=None)
    for h in hooks:
        h.remove()
    replaced = 0
    for parent in list(codec.encoder.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, MimiConv1d):
                setattr(parent, name, StaticMimiConv1d(child, lengths[id(child)]))
                replaced += 1
    print(f"static-pad MimiConv1d replaced: {replaced}")
    codec._ref_codes = ref_codes

    graphs = {
        "lm_prefill": Prefill(model.backbone, SEQ),
        "lm_decode": Decode(model.backbone, CAP),
        "depth_prefill": DepthPrefill(model.decoder),
        "depth_step": DepthStep(model.decoder),
        "codec_enc": CodecEncoder(codec, CODEC_SAMPLES),
        "codec_dec": CodecDecoder(codec.decoder),
    }
    for g in graphs.values():
        g.eval()
    if not args.check:
        h = torch.randn(1, SEQ, 512)
        cos, sin = graphs["lm_prefill"].cos, graphs["lm_prefill"].sin
        export_graph(graphs["lm_prefill"], (h, cos, sin), ONNX_DIR / "lm_prefill.onnx",
                     ["hidden", "cos", "sin"], ["hidden_out", "kv"])
        cap = CAP
        export_graph(graphs["lm_decode"],
                     (torch.randn(1, 1, 512), torch.randn(1, 1, 1, 64), torch.randn(1, 1, 1, 64),
                      torch.zeros(1, 1, 1, cap + 1), torch.zeros(len(model.backbone.layers), 2, 1, 2, cap, 64)),
                     ONNX_DIR / "lm_decode.onnx",
                     ["hidden", "cos", "sin", "mask", "past"], ["hidden_out", "kv"])
        export_graph(graphs["depth_prefill"], (torch.randn(1, 2, 512),),
                     ONNX_DIR / "depth_prefill.onnx", ["local"], ["hidden_out", "kv"])
        slots = DEPTH_SLOTS
        export_graph(graphs["depth_step"],
                     (torch.randn(1, 1, 512), torch.randn(1, 1, 1, 64), torch.randn(1, 1, 1, 64),
                      torch.zeros(1, 1, 1, slots + 1),
                      torch.zeros(len(model.decoder.layers), 2, 1, 2, slots, 64)),
                     ONNX_DIR / "depth_step.onnx",
                     ["local", "cos", "sin", "mask", "past"], ["hidden_out", "kv"])
        if not args.skip_codec:
            export_graph(graphs["codec_enc"], (torch.zeros(1, CODEC_SAMPLES),),
                         ONNX_DIR / "codec_encoder.onnx", ["audio"], ["codes"])
            dec = graphs["codec_dec"]
            inputs = [torch.zeros(1, 512, CODEC_FRAMES),
                      torch.zeros(dec.layers, 2, 1, dec.kv_heads, dec.window - 1, dec.dim),
                      torch.zeros(1, 1, CODEC_FRAMES, 64), torch.zeros(1, 1, CODEC_FRAMES, 64),
                      torch.zeros(1, 1, CODEC_FRAMES, dec.window - 1 + CODEC_FRAMES),
                      *dec.empty_states()]
            names = ["latent", "past", "cos", "sin", "mask"] + [f"conv_{i}" for i in range(len(dec.specs))]
            outputs = ["audio", "past_out"] + [f"conv_out_{i}" for i in range(len(dec.specs))]
            export_graph(dec, tuple(inputs), ONNX_DIR / "codec_decoder.onnx", names, outputs)
        meta = dict(seq=SEQ, cap=CAP, depth_slots=DEPTH_SLOTS, codec_frames=CODEC_FRAMES,
                    codec_samples=CODEC_SAMPLES, codec_states=graphs["codec_dec"].specs,
                    codec_layers=graphs["codec_dec"].layers,
                    codec_kv_heads=graphs["codec_dec"].kv_heads, codec_head_dim=graphs["codec_dec"].dim,
                    codec_window=graphs["codec_dec"].window)
        (ONNX_DIR / "graph_meta.json").write_text(json.dumps(meta, indent=2))

    print("validating ...", flush=True)
    run_checks(model, codec, graphs, with_codec=not args.skip_codec)
    print("saving assets ...", flush=True)
    cfg = C.export_assets(model, tokenizer, codec)
    print(json.dumps(cfg, indent=2))
    (ONNX_DIR / "checks.json").write_text(json.dumps(CHECKS, indent=2))
    for path in sorted(ONNX_DIR.glob("*.onnx")):
        print(f"{path.name:22s} {path.stat().st_size/1e6:8.1f} MB")
    print("done")


if __name__ == "__main__":
    main()
