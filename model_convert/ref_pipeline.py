#!/usr/bin/env python3
"""FP32 reference pipeline over the six ONNX graphs (host sampling in numpy).

This is the numerical ground truth for calibration, board accuracy checks and
the FP32 listening sample. It mirrors infer_onnx.py but with the split graphs:

  codec_enc -> prompt codes -> host RVQ -> lm_prefill -> loop(lm_decode,
  depth_prefill, depth_step, host sampling/heads) -> host RVQ -> codec_dec

    python export/ref_pipeline.py --text "..." --prompt-wav x.wav \
        --prompt-text "..." --out scratch/ref.wav [--record scratch/eval_io]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "export"))
import common as C
from common import CAP, CODEC_FRAMES, CODEC_SAMPLES, DEPTH_SLOTS, SEQ


def load_sessions(onnx_dir, threads=8):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    names = ("lm_prefill", "lm_decode", "depth_prefill", "depth_step",
             "codec_encoder", "codec_decoder")
    return {n: ort.InferenceSession(str(Path(onnx_dir) / f"{n}.onnx"), sess_options=opts,
                                    providers=["CPUExecutionProvider"]) for n in names}


class RefPipeline:
    def __init__(self, onnx_dir=ROOT / "export" / "onnx", assets=ROOT / "export" / "assets",
                 threads=8):
        from tokenizers import Tokenizer
        self.sess = load_sessions(onnx_dir, threads)
        self.cfg = json.loads((Path(assets) / "config.json").read_text())
        self.tokenizer = Tokenizer.from_file(str(Path(assets) / "tokenizer.json"))
        self.text_emb = np.load(Path(assets) / "text_embeddings.npy", mmap_mode="r")
        self.audio_emb = np.load(Path(assets) / "audio_embeddings.npy", mmap_mode="r")
        self.head0 = np.load(Path(assets) / "head0.npy")
        self.audio_head = np.load(Path(assets) / "audio_head.npy")
        self.proj = np.load(Path(assets) / "projection.npy")
        self.binv = np.load(Path(assets) / "backbone_inv_freq.npy")
        self.dinv = np.load(Path(assets) / "decoder_inv_freq.npy")
        z = np.load(Path(assets) / "codec_rvq.npz")
        self.book_sem = z["book_sem"][0]
        self.book_rest = z["book_rest"]
        self.ps_w, self.ps_b = z["proj_sem_w"], z["proj_sem_b"]
        self.pr_w, self.pr_b = z["proj_rest_w"], z["proj_rest_b"]
        self.offsets = np.arange(16, dtype=np.int64) * self.cfg["audio_vocab"]
        self.hop = self.cfg["hop_length"]
        self.sr = self.cfg["sample_rate"]
        self.cinv = np.load(Path(assets) / "codec_inv_freq.npy")
        self.evq = C.EncoderVQNP(Path(assets) / "codec_enc_vq.npz")
        self.cmeta = json.loads((Path(onnx_dir) / "graph_meta.json").read_text())
        self.cstates = [s["shape"] for s in self.cmeta["codec_states"]]
        self.stats = {}

    # ---------------------------------------------------------------- helpers
    def _run(self, name, feed):
        return self.sess[name].run(None, feed)

    def encode_reference(self, audio, record=None):
        audio = np.ascontiguousarray(audio[:CODEC_SAMPLES], np.float32)
        padded = np.zeros(CODEC_SAMPLES, np.float32)
        padded[:len(audio)] = audio
        feat = self._run("codec_encoder", {"audio": padded[None]})[0]
        codes = self.evq.encode(feat[0])
        frames = max(1, int(np.ceil(len(audio) / self.hop)))
        codes = codes[:, :frames].T.copy()
        if record is not None:
            record.setdefault("codec_encoder", []).append(dict(audio=padded[None], codes=codes[None]))
        return codes

    def codec_latent(self, codes):
        codes = np.asarray(codes, np.int64)
        sem = self.book_sem[codes[:, 0]] @ self.ps_w.T + self.ps_b
        acc = self.book_rest[0][codes[:, 1]].copy()
        for i in range(1, 15):
            acc += self.book_rest[i][codes[:, i + 1]]
        rest = acc @ self.pr_w.T + self.pr_b
        return np.ascontiguousarray((sem + rest).T, np.float32)

    @staticmethod
    def _sample(logits, noise, allow_eos):
        out = logits.astype(np.float32).copy()
        if not allow_eos:
            out[0] = -1e5
        out /= 0.9
        cutoff = np.partition(out, -10)[-10]
        out[out < cutoff] = -np.inf
        np.subtract(out, out.max(), out=out)
        np.exp(out, out=out)
        out /= out.sum()
        return int(np.argmax(out / noise))

    def _local_rope(self, pos):
        cos, sin = C.rope_cos_sin(self.binv, self.cfg["backbone_rope_scale"], [pos])
        return cos.reshape(1, 1, 1, -1), sin.reshape(1, 1, 1, -1)

    def _depth_rope(self, pos):
        cos, sin = C.rope_cos_sin(self.dinv, self.cfg["decoder_rope_scale"], [pos])
        return cos.reshape(1, 1, 1, -1), sin.reshape(1, 1, 1, -1)

    def _codec_mask(self, offset, n):
        key_abs = np.arange(self.cfg["window"] - 1 + n) + offset - (self.cfg["window"] - 1)
        pos_abs = offset + np.arange(n)
        allowed = ((key_abs[None, :] <= pos_abs[:, None]) & (key_abs[None, :] >= 0) &
                   (key_abs[None, :] >= pos_abs[:, None] - (self.cfg["window"] - 1)))
        return np.where(allowed, 0.0, -30.0).astype(np.float32)[None, None]

    # ------------------------------------------------------------- generation
    def generate(self, text, prompt_audio, prompt_text, lang="Chinese", seed=1234,
                 max_frames=400, record=None, record_every=8):
        started = time.perf_counter()
        prompt = self.encode_reference(prompt_audio, record=record)  # [T,16]
        sentence = C.conditioned_text(prompt_text.strip() + text.strip(), lang)
        ids = np.array(self.tokenizer.encode(sentence).ids, np.int64)
        prefix = self.audio_emb[prompt + 1 + self.offsets].sum(axis=1).astype(np.float32)
        hidden = np.concatenate((self.text_emb[ids], prefix), axis=0)
        real = hidden.shape[0]
        if real >= SEQ:
            raise ValueError(f"prompt too long: {real} tokens >= {SEQ}")
        hidden_pad = np.zeros((SEQ, hidden.shape[1]), np.float32)
        hidden_pad[:real] = hidden
        hidden_pad[real:] = hidden[-1]
        cos_p, sin_p = C.rope_cos_sin(self.binv, self.cfg["backbone_rope_scale"], np.arange(SEQ))
        if record is not None:
            record.setdefault("lm_prefill", []).append(
                dict(hidden=hidden_pad[None], cos=cos_p.reshape(1, 1, SEQ, -1),
                     sin=sin_p.reshape(1, 1, SEQ, -1)))
        kv_out, kv = self._run("lm_prefill", {"hidden": hidden_pad[None],
                                              "cos": cos_p.reshape(1, 1, SEQ, -1),
                                              "sin": sin_p.reshape(1, 1, SEQ, -1)})
        last = kv_out[0, real - 1]
        cache = np.zeros((self.cfg["backbone_layers"], 2, 1,
                          self.cfg["backbone_kv_heads"], CAP, 64), np.float32)
        cache[:, :, :, :, :real] = kv[:, :, :, :, :real]
        pos = real
        rng = np.random.default_rng(seed)
        generated = []
        eos = False
        depth_buf = np.zeros((self.cfg["decoder_layers"], 2, 1,
                              self.cfg["decoder_kv_heads"], DEPTH_SLOTS, 64), np.float32)
        depth_mask = np.full((1, 1, 1, DEPTH_SLOTS + 1), -30.0, np.float32)
        decoder_cache_full = np.zeros((self.cfg["backbone_layers"], 2, 1,
                                       self.cfg["backbone_kv_heads"], CAP, 64), np.float32)
        decoder_cache_full[:, :, :, :, :real] = cache[:, :, :, :, :real]

        for frame in range(max_frames):
            if pos >= CAP:
                break
            noise = rng.exponential(size=(16, self.cfg["audio_vocab"])).astype(np.float32)
            np.maximum(noise, np.finfo(np.float32).tiny, out=noise)
            logits0 = self.head0 @ last
            c0 = self._sample(logits0, noise[0], frame > self.cfg["min_frames"])
            local = np.concatenate(((last @ self.proj.T)[None],
                                    (self.audio_emb[c0] @ self.proj.T)[None]), axis=0)[None]
            dh, dkv = self._run("depth_prefill", {"local": local})
            depth_buf[:, :, :, :, :2] = dkv
            if record is not None and frame % record_every == 0:
                record.setdefault("depth_prefill", []).append(dict(local=local))
            codes = [c0]
            h = dh[0]
            for level in range(15):
                logits = h @ self.audio_head[level]
                ci = self._sample(logits, noise[level + 1], True)
                codes.append(ci)
                if level == 14:
                    break
                token_pos = level + 2
                local = (self.audio_emb[ci + (level + 1) * self.cfg["audio_vocab"]] @ self.proj.T)[None, None]
                depth_mask[...] = -30.0
                depth_mask[..., :token_pos] = 0.0
                depth_mask[..., -1] = 0.0
                if record is not None and frame % record_every == 0:
                    record.setdefault("depth_step", []).append(
                        dict(local=local, mask=depth_mask.copy(), past=depth_buf.copy(),
                             cos=self._depth_rope(token_pos)[0], sin=self._depth_rope(token_pos)[1]))
                h_out, newkv = self._run("depth_step", {
                    "local": local, "mask": depth_mask,
                    "cos": self._depth_rope(token_pos)[0], "sin": self._depth_rope(token_pos)[1],
                    "past": depth_buf})
                h = h_out[0]
                depth_buf[:, :, :, :, token_pos] = newkv[:, :, :, :, 0]
            codes = np.array(codes, np.int64)
            if np.all(codes == 0):
                eos = True
                break
            generated.append(np.clip(codes - 1, 0, self.cfg["audio_vocab"] - 1 - 1))
            next_hidden = self.audio_emb[codes + self.offsets].sum(axis=0)[None]
            bmask = np.full((1, 1, 1, CAP + 1), -30.0, np.float32)
            bmask[..., :pos] = 0.0
            bmask[..., -1] = 0.0
            if record is not None and frame % record_every == 0 and len(record.get('lm_decode', [])) < 90:
                record.setdefault("lm_decode", []).append(
                    dict(hidden=next_hidden[None], mask=bmask.copy(), past=decoder_cache_full.copy(),
                         cos=self._local_rope(pos)[0], sin=self._local_rope(pos)[1]))
            out, newkv = self._run("lm_decode", {"hidden": next_hidden[None], "mask": bmask,
                                                  "cos": self._local_rope(pos)[0],
                                                  "sin": self._local_rope(pos)[1],
                                                  "past": decoder_cache_full})
            decoder_cache_full[:, :, :, :, pos] = newkv[:, :, :, :, 0]
            last = out[0]
            pos += 1

        all_codes = np.concatenate((prompt, np.stack(generated) if generated else np.zeros((0, 16), np.int64)))
        wav = self.decode_codes(all_codes, record=record)
        wav = wav[len(prompt) * self.hop:].astype(np.float32)
        self.stats = dict(frames=len(generated), eos=eos, prompt_frames=len(prompt),
                          tokens=real, seconds=len(wav) / self.sr,
                          elapsed=time.perf_counter() - started)
        self.generated_codes = np.stack(generated) if generated else np.zeros((0, 16), np.int64)
        return wav

    def decode_codes(self, codes, record=None, record_every=3):
        return self.decode_latent(self.codec_latent(codes)[None], record=record, record_every=record_every)

    def decode_latent(self, latent, record=None, record_every=3):
        total = latent.shape[2]
        past = np.zeros((self.cmeta["codec_layers"], 2, 1, self.cmeta["codec_kv_heads"],
                         self.cfg["window"] - 1, self.cmeta["codec_head_dim"]), np.float32)
        states = [np.zeros(shape, np.float32) for shape in self.cstates]
        outs = []
        offset = 0
        for start in range(0, total, CODEC_FRAMES):
            n = min(CODEC_FRAMES, total - start)
            chunk = np.zeros((1, latent.shape[1], CODEC_FRAMES), np.float32)
            chunk[:, :, :n] = latent[:, :, start:start + n]
            cos, sin = C.rope_cos_sin(self.cinv, 1.0, np.arange(offset, offset + CODEC_FRAMES))
            feed = {"latent": chunk, "past": past, "cos": cos.reshape(1, 1, -1, 64),
                    "sin": sin.reshape(1, 1, -1, 64), "mask": self._codec_mask(offset, CODEC_FRAMES)}
            for i, s in enumerate(states):
                feed[f"conv_{i}"] = s
            if record is not None and start % (record_every * CODEC_FRAMES) == 0:
                record.setdefault("codec_decoder", []).append(dict(feed))
            out = self._run("codec_decoder", feed)
            outs.append(out[0])
            past = out[1]
            states = out[2:]
            offset += CODEC_FRAMES
        wav = np.concatenate(outs, axis=-1).reshape(-1)
        return wav[:total * self.hop]



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--prompt-wav", required=True)
    ap.add_argument("--prompt-text", required=True)
    ap.add_argument("--lang", default="Chinese")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", default=None)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    pipe = RefPipeline(threads=args.threads)
    audio = C.read_reference(args.prompt_wav)
    record = {} if args.record else None
    wav = pipe.generate(args.text, audio, args.prompt_text, lang=args.lang, seed=args.seed,
                        max_frames=args.max_frames, record=record)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    wav = np.ascontiguousarray(np.clip(wav, -1, 1).astype(np.float32))
    print("diag", type(pipe.sr), pipe.sr, wav.shape, wav.dtype, bool(np.isfinite(wav).all()), flush=True)
    tmp = str(out) + ".tmp.wav"
    sf.write(tmp, wav, int(pipe.sr), subtype="PCM_16", format="WAV")
    import os
    os.replace(tmp, out)
    np.save(out.with_suffix(".codes.npy"), pipe.generated_codes)
    print(json.dumps(dict(out=str(out), **pipe.stats), indent=2))
    if record:
        for name, items in record.items():
            print(f"recorded {name}: {len(items)}")
        np.save(out.with_suffix(".record.npy"), np.array([1]))  # placeholder marker
        json.dump({k: len(v) for k, v in record.items()}, open(str(out) + ".record.json", "w"))


if __name__ == "__main__":
    main()
