"""Shared loaders / host-side pieces for the M5-TTS AX650 export."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
M5_SRC = Path(os.environ.get("M5_SRC", ROOT / "origin" / "M5-TTS"))
for extra in (str(ROOT / ".pylib"), str(M5_SRC), str(M5_SRC / "onnx")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

LM_BUNDLE = Path(os.environ.get("M5_LM_BUNDLE", ROOT.parents[1] / "M5-TTS_checkpoints"))
CODEC_BUNDLE = Path(os.environ.get("M5_CODEC_BUNDLE", ROOT / "origin" / "codec"))
ONNX_DIR = ROOT / "export" / "onnx"
ASSET_DIR = ROOT / "export" / "assets"

# Static deployment shapes
SEQ = int(os.environ.get("M5_SEQ", 256))          # prefill tokens
CAP = int(os.environ.get("M5_CAP", 512))          # total backbone KV slots
DEPTH_SLOTS = 16                                   # 2 prefill + 14 steps
CODEC_FRAMES = int(os.environ.get("M5_CODEC_FRAMES", 4))   # 320 ms per call
CODEC_SAMPLES = 115200                             # 4.8 s reference window


def load_lm():
    import torch
    from checkpoint import load_checkpoint
    model, tokenizer = load_checkpoint(LM_BUNDLE)
    return model.eval().float(), tokenizer


def load_codec():
    import torch
    from m5.codec_qwen12hz import Qwen3TTSTokenizer
    tokenizer = Qwen3TTSTokenizer.from_pretrained(str(CODEC_BUNDLE), dtype=torch.float32,
                                                 local_files_only=True)
    return tokenizer.model.eval().float()


def rope_cos_sin(inv_freq, scaling, positions):
    """Exactly torch's inv_freq @ pos -> cat -> cos/sin for one or many positions."""
    freqs = np.asarray(positions, np.float32)[:, None] * np.asarray(inv_freq, np.float32)[None, :]
    emb = np.concatenate((freqs, freqs), axis=-1)
    return (np.cos(emb) * scaling).astype(np.float32), (np.sin(emb) * scaling).astype(np.float32)


class EncoderVQ:
    """Host-side codec encoder RVQ (ArgMin is not supported by Pulsar2)."""

    def __init__(self, codec):
        quant = codec.encoder.quantizer
        self.sem_w = quant.semantic_residual_vector_quantizer.input_proj.weight.detach().numpy()[:, :, 0].astype(np.float32)
        self.ac_w = quant.acoustic_residual_vector_quantizer.input_proj.weight.detach().numpy()[:, :, 0].astype(np.float32)

        def books(branch):
            out = []
            for layer in branch.layers:
                cb = layer.codebook
                out.append((cb.embed_sum / cb.cluster_usage.clamp(min=cb.epsilon)[:, None]).detach().numpy().astype(np.float32))
            return np.stack(out)

        self.sem_book = books(quant.semantic_residual_vector_quantizer)
        self.ac_book = books(quant.acoustic_residual_vector_quantizer)

    @staticmethod
    def _nearest(x, book):
        # squared euclidean distance, x [T,D] book [K,D]
        x2 = (x * x).sum(1)[:, None]
        b2 = (book * book).sum(1)[None, :]
        d = x2 + b2 - 2.0 * (x @ book.T)
        return np.argmin(d, axis=1)

    def encode(self, feat, num_quantizers=16):
        x = np.asarray(feat, np.float32)
        if x.shape[0] == 512 and x.shape[1] != 512:
            x = x.T  # [T,512]
        out = []
        res = x @ self.sem_w.T
        for book in self.sem_book[:1]:
            idx = self._nearest(res, book)
            out.append(idx)
            res = res - book[idx]
        res = x @ self.ac_w.T
        for book in self.ac_book[:num_quantizers - 1]:
            idx = self._nearest(res, book)
            out.append(idx)
            res = res - book[idx]
        return np.stack(out)  # [K, T]

    def tables(self):
        return dict(enc_sem_w=self.sem_w, enc_ac_w=self.ac_w,
                    enc_sem_book=self.sem_book[:1], enc_ac_book=self.ac_book[:15])


class EncoderVQNP:
    """Numpy-only encoder RVQ from the exported tables."""

    def __init__(self, path):
        z = np.load(path)
        self.sem_w = z["enc_sem_w"]
        self.ac_w = z["enc_ac_w"]
        self.sem_book = z["enc_sem_book"]
        self.ac_book = z["enc_ac_book"]

    @staticmethod
    def _nearest(x, book):
        x2 = (x * x).sum(1)[:, None]
        b2 = (book * book).sum(1)[None, :]
        d = x2 + b2 - 2.0 * (x @ book.T)
        return np.argmin(d, axis=1)

    def encode(self, feat, num_quantizers=16):
        x = np.asarray(feat, np.float32)
        if x.shape[0] == 512 and x.shape[1] != 512:
            x = x.T
        out = []
        res = x @ self.sem_w.T
        book = self.sem_book[0]
        idx = self._nearest(res, book)
        out.append(idx)
        res = res - book[idx]
        res = x @ self.ac_w.T
        for book in self.ac_book[:num_quantizers - 1]:
            idx = self._nearest(res, book)
            out.append(idx)
            res = res - book[idx]
        return np.stack(out)


class RVQ:
    """Host-side quantizer.decode (16 codebooks -> latent [512, T])."""

    def __init__(self, decoder):
        self.sem = decoder.quantizer.rvq_first
        self.rest = decoder.quantizer.rvq_rest

    @staticmethod
    def _branch(branch, codes):
        # codes [K, T] -> latent [512, T]
        total = None
        for index, quantizer in enumerate(branch.vq.layers):
            book = quantizer._codebook
            table = (book.embedding_sum / book.cluster_usage.clamp(min=book.epsilon)[:, None])
            if branch.vq.layers[index] is None:
                continue
            emb = table.detach().cpu().numpy()[codes[index]]          # [T,256]
            total = emb if total is None else total + emb
        weight = branch.output_proj.weight.detach().cpu().numpy()[:, :, 0]   # [512,256]
        out = total @ weight.T
        if branch.output_proj.bias is not None:
            out = out + branch.output_proj.bias.detach().cpu().numpy()
        return out.T.astype(np.float32)

    def decode(self, codes):
        codes = np.asarray(codes, np.int64)
        return self._branch(self.sem, codes[:1]) + self._branch(self.rest, codes[1:])

    def tables(self):
        out = {}
        for name, branch in (("sem", self.sem), ("rest", self.rest)):
            books = []
            for quantizer in branch.vq.layers:
                book = quantizer._codebook
                books.append((book.embedding_sum / book.cluster_usage.clamp(min=book.epsilon)[:, None])
                             .detach().cpu().numpy().astype(np.float32))
            out[f"book_{name}"] = np.stack(books)
            out[f"proj_{name}_w"] = branch.output_proj.weight.detach().cpu().numpy()[:, :, 0].astype(np.float32)
            out[f"proj_{name}_b"] = (branch.output_proj.bias.detach().cpu().numpy().astype(np.float32)
                                     if branch.output_proj.bias is not None else np.zeros(512, np.float32))
        return out


def conditioned_text(text, lang):
    text = (text or "").strip()
    return ("<|caption_begin|><|caption_end|><|accent_begin|><|accent_end|>"
            "<|speaker_start|>None<|speaker_end|>"
            f"<|lang_begin|>{'' if lang is None else lang}<|lang_end|>{text}")


def read_reference(path, sample_rate=24000):
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        divisor = gcd(sr, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, sr // divisor).astype(np.float32)
    audio /= np.max(np.abs(audio)) + 1e-8
    threshold = 0.02 if len(audio) > 10 * sample_rate else 0.015
    speech = np.flatnonzero(np.abs(audio) > threshold)
    if len(speech):
        cuts = np.flatnonzero(np.diff(speech) > int(0.15 * sample_rate))
        starts = np.r_[speech[0], speech[cuts + 1]]
        final_end = speech[-1] + 1 if len(audio) - speech[-1] - 1 >= int(0.15 * sample_rate) else len(audio)
        ends = np.r_[speech[cuts] + 1, final_end]
        audio = np.concatenate([audio[max(0, a - 1200):min(len(audio), b + 1200)]
                                for a, b in zip(starts, ends)])
    return np.ascontiguousarray(audio, dtype=np.float32)


def sample_topk(logits, topk=10, temperature=0.9, noise=None, allow_eos=True):
    logits = logits.astype(np.float32).copy()
    if not allow_eos:
        logits[0] = -1e5
    logits = logits / temperature
    cutoff = np.partition(logits, -topk)[-topk]
    logits[logits < cutoff] = -np.inf
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return int(np.argmax(probs / noise))


def export_assets(model, tokenizer, codec, outdir=ASSET_DIR):
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "text_embeddings.npy", model.text_embeddings.weight.detach().numpy().astype(np.float32))
    np.save(outdir / "audio_embeddings.npy", model.audio_embeddings.weight.detach().numpy().astype(np.float32))
    proj = model.projection.weight.detach().numpy().astype(np.float32)
    np.save(outdir / "projection.npy", proj)
    np.save(outdir / "projected_audio.npy",
            (model.audio_embeddings.weight.detach().numpy() @ proj.T).astype(np.float32))
    np.save(outdir / "head0.npy", model.codebook0_head.weight.detach().numpy().astype(np.float32))
    np.save(outdir / "audio_head.npy", model.audio_head.detach().numpy().astype(np.float32))
    np.save(outdir / "backbone_inv_freq.npy", model.backbone.position_embedding.inv_freq.detach().numpy().astype(np.float32))
    np.save(outdir / "decoder_inv_freq.npy", model.decoder.position_embedding.inv_freq.detach().numpy().astype(np.float32))
    np.save(outdir / "codec_inv_freq.npy", codec.decoder.pre_transformer.rotary_emb.inv_freq.detach().numpy().astype(np.float32))
    np.savez(outdir / "codec_enc_vq.npz", **EncoderVQ(codec).tables())
    rvq = RVQ(codec.decoder)
    np.savez(outdir / "codec_rvq.npz", **rvq.tables())
    tokenizer.backend_tokenizer.save(str(outdir / "tokenizer.json"))
    cfg = dict(seq=SEQ, cap=CAP, depth_slots=DEPTH_SLOTS, codec_frames=CODEC_FRAMES,
               codec_samples=CODEC_SAMPLES, codebooks=16, audio_vocab=2051,
               hidden=model.backbone.config.hidden_size,
               backbone_layers=len(model.backbone.layers),
               backbone_heads=model.backbone.config.num_attention_heads,
               backbone_kv_heads=model.backbone.config.num_key_value_heads,
               decoder_layers=len(model.decoder.layers),
               decoder_heads=model.decoder.config.num_attention_heads,
               decoder_kv_heads=model.decoder.config.num_key_value_heads,
               sample_rate=24000, hop_length=1920, topk=10, temperature=0.9,
               min_frames=6, window=72,
               backbone_rope_scale=float(model.backbone.position_embedding.attention_scaling),
               decoder_rope_scale=float(model.decoder.position_embedding.attention_scaling),
               eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id)
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))
    meta_path = ROOT / "export" / "onnx" / "graph_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        (outdir / "codec_meta.json").write_text(json.dumps(meta, indent=2))
    return cfg
