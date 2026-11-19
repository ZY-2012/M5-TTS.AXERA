"""M5-TTS AX650 runtime (numpy + axengine; no torch / onnxruntime needed).

    reference wav --codec_encoder--> prompt codes --host RVQ--> prompt embeds
    text ids + prompt embeds --lm_prefill--> hidden + KV
    loop per frame (12.5 fps):
        hidden --head0/sample--> c0
        [hidden, c0] --depth_prefill--> h + KV
        h --audio_head[i]/sample--> c_i --depth_step (x14)--> next h
        sum(audio_emb[codes]) --lm_decode--> next hidden
    prompt+generated codes --host RVQ--> latent --codec_decoder--> 24 kHz wav
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

NEG = -30.0


# ------------------------------------------------------------------ engines
class OnnxEngine:
    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def run(self, feed):
        return self.sess.run(self.output_names, {k: np.ascontiguousarray(feed[k]) for k in self.input_names})


class AxEngine:
    def __init__(self, path):
        import axengine as axe
        self.sess = axe.InferenceSession(str(path), providers=["AxEngineExecutionProvider"])
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def run(self, feed):
        return self.sess.run(self.output_names, {k: np.ascontiguousarray(feed[k], np.float32) for k in self.input_names})


# ------------------------------------------------------------------ helpers
def read_wav(path, sample_rate=24000):
    import soundfile as sf
    from math import gcd
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        from scipy.signal import resample_poly
        g = gcd(sr, sample_rate)
        audio = resample_poly(audio, sample_rate // g, sr // g).astype(np.float32)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    threshold = 0.02 if len(audio) > 10 * sample_rate else 0.015
    speech = np.flatnonzero(np.abs(audio) > threshold)
    if len(speech):
        cuts = np.flatnonzero(np.diff(speech) > int(0.15 * sample_rate))
        starts = np.r_[speech[0], speech[cuts + 1]]
        final_end = speech[-1] + 1 if len(audio) - speech[-1] - 1 >= int(0.15 * sample_rate) else len(audio)
        ends = np.r_[speech[cuts] + 1, final_end]
        audio = np.concatenate([audio[max(0, a - 1200):min(len(audio), b + 1200)]
                                for a, b in zip(starts, ends)])
    return np.ascontiguousarray(audio, np.float32)


def write_wav(path, wav, sample_rate=24000):
    import soundfile as sf
    sf.write(str(path), np.clip(wav, -1, 1).astype(np.float32), int(sample_rate),
             subtype="PCM_16", format="WAV")


def rope_tables(inv_freq, scale, positions):
    freqs = np.asarray(positions, np.float32)[:, None] * np.asarray(inv_freq, np.float32)[None, :]
    emb = np.concatenate((freqs, freqs), axis=-1)
    return (np.cos(emb) * scale).astype(np.float32), (np.sin(emb) * scale).astype(np.float32)


def conditioned_text(text, lang):
    text = (text or "").strip()
    return ("<|caption_begin|><|caption_end|><|accent_begin|><|accent_end|>"
            "<|speaker_start|>None<|speaker_end|>"
            f"<|lang_begin|>{'' if lang is None else lang}<|lang_end|>{text}")


class M5AX:
    def __init__(self, model_dir, assets_dir, engine="ax", threads=4):
        self.assets_dir = str(assets_dir)
        self.cfg = json.loads((Path(assets_dir) / "config.json").read_text())
        self.cap = self.cfg["cap"]
        self.seq = self.cfg["seq"]
        self.hop = self.cfg["hop_length"]
        self.sr = self.cfg["sample_rate"]
        self.noise_tiny = np.finfo(np.float32).tiny
        engine_cls = AxEngine if engine == "ax" else OnnxEngine
        suffix = ".axmodel" if engine == "ax" else ".onnx"
        self.engines = {}
        for name in ("lm_prefill", "lm_decode", "depth_prefill", "depth_step",
                     "codec_encoder", "codec_decoder"):
            self.engines[name] = engine_cls(Path(model_dir) / f"{name}{suffix}")
        a = Path(assets_dir)
        self.text_emb = np.load(a / "text_embeddings.npy", mmap_mode="r")
        self.audio_emb = np.load(a / "audio_embeddings.npy", mmap_mode="r")
        self.head0 = np.load(a / "head0.npy")
        self.audio_head = np.load(a / "audio_head.npy")
        self.proj = np.load(a / "projection.npy")
        self.binv = np.load(a / "backbone_inv_freq.npy")
        self.dinv = np.load(a / "decoder_inv_freq.npy")
        self.cinv = np.load(a / "codec_inv_freq.npy")
        zq = np.load(a / "codec_enc_vq.npz")
        self.evq_sem_w, self.evq_ac_w = zq["enc_sem_w"], zq["enc_ac_w"]
        self.evq_sem_book, self.evq_ac_book = zq["enc_sem_book"], zq["enc_ac_book"]
        z = np.load(a / "codec_rvq.npz")
        self.book_sem = z["book_sem"][0]
        self.book_rest = z["book_rest"]
        self.ps_w, self.ps_b = z["proj_sem_w"], z["proj_sem_b"]
        self.pr_w, self.pr_b = z["proj_rest_w"], z["proj_rest_b"]
        meta = json.loads((a / "codec_meta.json").read_text())
        self.clayers = meta["codec_layers"]
        self.ckv = meta["codec_kv_heads"]
        self.cdim = meta["codec_head_dim"]
        self.cwindow = meta["codec_window"]
        self.cstates = [s["shape"] for s in meta["codec_states"]]
        self.cframes = meta["codec_frames"]
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(a / "tokenizer.json"))
        self.offsets = np.arange(16, dtype=np.int64) * self.cfg["audio_vocab"]
        self.last_timing = {}
        self.generated_codes = None
        self._tables_ready = False
        self._ref_cache = {}
        self._prompt_cache = {}
        prior_path = a / "codebook_prior.npy"
        code_path = a / "mean_code.npy"
        self.prior = np.load(prior_path) if prior_path.exists() else np.zeros((16, 512), np.float32)
        self.mean_code = (np.load(code_path).astype(np.int64) if code_path.exists()
                          else np.arange(16, dtype=np.int64))
        # precomputed RoPE tables (tokens are <= cap + 4)
        top = self.cap + 8
        cos_b, sin_b = rope_tables(self.binv, self.cfg["backbone_rope_scale"], np.arange(top))
        self.bcos = cos_b.reshape(top, 64)
        self.bsin = sin_b.reshape(top, 64)
        cos_d, sin_d = rope_tables(self.dinv, self.cfg["decoder_rope_scale"], np.arange(self.cfg["depth_slots"] + 2))
        self.dcos = cos_d.reshape(-1, 64)
        self.dsin = sin_d.reshape(-1, 64)
        ctop = 2048
        cos_c, sin_c = rope_tables(self.cinv, 1.0, np.arange(ctop))
        self.ccos = cos_c.reshape(ctop, 64)
        self.csin = sin_c.reshape(ctop, 64)


    # -------------------------------------------------------------- pieces
    def _prepare_tables(self):
        if self._tables_ready:
            return
        path = Path(self.assets_dir) / "projected_audio.npy"
        if path.is_file():
            self.proj_audio = np.load(path, mmap_mode="r")
        else:  # fallback: project every audio embedding row, fp32 [16*2051, 512]
            self.proj_audio = np.asarray(self.audio_emb @ self.proj.T, np.float32)
        self._tables_ready = True

    def _run(self, name, feed):
        return self.engines[name].run(feed)

    @staticmethod
    def _nearest(x, book):
        x2 = (x * x).sum(1)[:, None]
        b2 = (book * book).sum(1)[None, :]
        d = x2 + b2 - 2.0 * (x @ book.T)
        return np.argmin(d, axis=1)

    def encode_reference(self, audio, key=None):
        if key is not None and key in self._ref_cache:
            return self._ref_cache[key].copy()
        codes = self._encode_reference(audio)
        if key is not None:
            if len(self._ref_cache) > 4:
                self._ref_cache.pop(next(iter(self._ref_cache)))
            self._ref_cache[key] = codes
        return codes

    def _encode_reference(self, audio):
        padded = np.zeros(self.cfg["codec_samples"], np.float32)
        audio = audio[:self.cfg["codec_samples"]]
        padded[:len(audio)] = audio
        feat = self._run("codec_encoder", {"audio": padded[None]})[0]   # [1,512,60]
        x = feat[0].T
        res = x @ self.evq_sem_w.T
        book = self.evq_sem_book[0]
        idx = self._nearest(res, book)
        codes = [idx]
        res = res - book[idx]
        res = x @ self.evq_ac_w.T
        for book in self.evq_ac_book[:15]:
            idx = self._nearest(res, book)
            codes.append(idx)
            res = res - book[idx]
        codes = np.stack(codes)
        frames = max(1, int(math.ceil(len(audio) / self.hop)))
        return codes[:, :frames].T.copy()

    def codec_latent(self, codes, keep=16):
        codes = np.asarray(codes, np.int64)
        sem = self.book_sem[codes[:, 0]] @ self.ps_w.T + self.ps_b
        acc = self.book_rest[0][codes[:, 1]].copy()
        for i in range(1, max(0, keep - 1)):
            acc += self.book_rest[i][codes[:, i + 1]]
        rest = acc @ self.pr_w.T + self.pr_b
        if keep < 16:
            rest = rest + self.prior[keep:].sum(axis=0)   # average residual of dropped codebooks
        return np.ascontiguousarray((sem + rest).T, np.float32)

    def _sample(self, logits, noise, allow_eos, topk=10, temperature=0.9, greedy=False):
        if greedy:
            idx = np.argpartition(logits, -topk)[-topk:]
            best = idx[np.argmax(logits[idx])]
            if not allow_eos and best == 0:
                logits = logits.copy()
                logits[0] = -1e5
                idx = np.argpartition(logits, -topk)[-topk:]
                best = idx[np.argmax(logits[idx])]
            return int(best)
        idx = np.argpartition(logits, -topk)[-topk:]
        vals = logits[idx] / temperature
        if not allow_eos and 0 in idx:
            vals[idx == 0] = -1e5
        vals = vals - vals.max()
        np.exp(vals, out=vals)
        vals /= vals.sum()
        pick = int(np.argmax(vals / noise[idx]))
        return int(idx[pick])

    def _head(self, h, level):
        return np.dot(h, self.audio_head[level])

    def _local_rope(self, pos):
        return self.bcos[pos].reshape(1, 1, 1, 64), self.bsin[pos].reshape(1, 1, 1, 64)

    def _depth_rope(self, pos):
        return self.dcos[pos].reshape(1, 1, 1, 64), self.dsin[pos].reshape(1, 1, 1, 64)

    def _codec_mask(self, offset, n):
        w = self.cwindow
        key_abs = np.arange(w - 1 + n) + offset - (w - 1)
        pos_abs = offset + np.arange(n)
        allowed = ((key_abs[None, :] <= pos_abs[:, None]) & (key_abs[None, :] >= 0) &
                   (key_abs[None, :] >= pos_abs[:, None] - (w - 1)))
        return np.where(allowed, 0.0, NEG).astype(np.float32)[None, None]

    # ----------------------------------------------------------- generation
    def warmup(self):
        t = self.last_timing
        try:
            self.encode_reference(np.zeros(2400, np.float32))
            self._run("lm_prefill", {
                "hidden": np.zeros((1, self.seq, self.cfg["hidden"]), np.float32),
                "cos": np.zeros((1, 1, self.seq, 64), np.float32),
                "sin": np.zeros((1, 1, self.seq, 64), np.float32)})
            self._run("lm_decode", {
                "hidden": np.zeros((1, 1, self.cfg["hidden"]), np.float32),
                "cos": np.zeros((1, 1, 1, 64), np.float32),
                "sin": np.zeros((1, 1, 1, 64), np.float32),
                "mask": np.zeros((1, 1, 1, self.cap + 1), np.float32),
                "past": np.zeros((self.cfg["backbone_layers"], 2, 1,
                                  self.cfg["backbone_kv_heads"], self.cap, 64), np.float32)})
            self._run("depth_prefill", {"local": np.zeros((1, 2, self.cfg["hidden"]), np.float32)})
            self._run("depth_step", {
                "local": np.zeros((1, 1, self.cfg["hidden"]), np.float32),
                "cos": np.zeros((1, 1, 1, 64), np.float32),
                "sin": np.zeros((1, 1, 1, 64), np.float32),
                "mask": np.zeros((1, 1, 1, self.cfg["depth_slots"] + 1), np.float32),
                "past": np.zeros((self.cfg["decoder_layers"], 2, 1,
                                  self.cfg["decoder_kv_heads"], self.cfg["depth_slots"], 64), np.float32)})
            self.decode_codes(np.zeros((self.cframes, 16), np.int64))
        finally:
            t.clear()

    # ------------------------------------------------------------ codec loop
    class _CodecStream:
        """Incremental codec decoder state (KV window + 21 conv states)."""

        def __init__(self, tts, keep=16):
            self.tts = tts
            self.keep = keep
            self.past = np.zeros((tts.clayers, 2, 1, tts.ckv, tts.cwindow - 1, tts.cdim), np.float32)
            self.states = [np.zeros(shape, np.float32) for shape in tts.cstates]
            self.offset = 0
            self.pending = np.zeros((0, 16), np.int64)

        def snapshot(self):
            return (self.past.copy(), [state.copy() for state in self.states],
                    self.offset, self.pending.copy())

        def clone(self):
            other = self.tts._CodecStream.__new__(self.tts._CodecStream)
            other.tts = self.tts
            other.past, states, other.offset, other.pending = self.snapshot()
            other.states = states
            return other

        def push(self, codes, final=False):
            self.pending = np.concatenate((self.pending, np.asarray(codes, np.int64)))
            frames = self.tts.cframes
            pcm = []
            while len(self.pending) >= frames or (final and len(self.pending)):
                block = self.pending[:frames]
                self.pending = self.pending[len(block):]
                pcm.append(self._chunk(block))
            return np.concatenate(pcm) if pcm else np.zeros(0, np.float32)

        def _chunk(self, codes):
            tts = self.tts
            n = len(codes)
            latent = tts.codec_latent(codes, keep=self.keep)[None]
            chunk = np.zeros((1, 512, tts.cframes), np.float32)
            chunk[:, :, :n] = latent
            off = self.offset
            feed = {"latent": chunk, "past": self.past,
                    "cos": tts.ccos[off:off + tts.cframes][None, None],
                    "sin": tts.csin[off:off + tts.cframes][None, None],
                    "mask": tts._codec_mask(off, tts.cframes)}
            for i, state in enumerate(self.states):
                feed[f"conv_{i}"] = state
            out = tts._run("codec_decoder", feed)
            self.past = out[1]
            self.states = out[2:]
            self.offset += n
            return out[0].reshape(-1)[:n * tts.hop]

    def decode_codes(self, codes, keep=16):
        stream = self._CodecStream(self, keep=keep)
        return stream.push(codes, final=True)

    # --------------------------------------------------------------- generation
    def _prepare(self, text, prompt_wav, prompt_text, lang, seed, max_frames, prompt_codes=None, codebooks=16):
        timing = {}
        t0 = time.perf_counter()
        key = None
        stream = None
        prefix = None
        cached = False
        if prompt_codes is None:
            if isinstance(prompt_wav, (str, Path)):
                stat = Path(prompt_wav).stat()
                key = (str(Path(prompt_wav).resolve()), stat.st_size, stat.st_mtime_ns, codebooks)
            entry = self._prompt_cache.get(key) if key is not None else None
            if entry is not None:
                prompt = entry["codes"].copy()
                prefix = entry["prefix"].copy()
                stream = entry["stream"].clone()
                cached = True
            else:
                audio = read_wav(prompt_wav, self.sr)
                prompt = self.encode_reference(audio, key=key)
        else:
            prompt = np.asarray(prompt_codes, np.int64)
        timing["codec_enc"] = time.perf_counter() - t0
        timing["ref_cached"] = cached
        sentence = conditioned_text(prompt_text.strip() + text.strip(), lang)
        ids = np.array(self.tokenizer.encode(sentence).ids, np.int64)
        if prefix is None:
            prefix = self.audio_emb[prompt + 1 + self.offsets].sum(axis=1).astype(np.float32)
        hidden = np.concatenate((self.text_emb[ids], prefix), axis=0)
        real = hidden.shape[0]
        if real >= self.seq:
            raise ValueError(f"prompt is {real} tokens, limit is {self.seq}")
        hidden_pad = np.zeros((self.seq, self.cfg["hidden"]), np.float32)
        hidden_pad[:real] = hidden
        hidden_pad[real:] = hidden[-1]
        cos_p = self.bcos[:self.seq].reshape(1, 1, self.seq, 64)
        sin_p = self.bsin[:self.seq].reshape(1, 1, self.seq, 64)
        t1 = time.perf_counter()
        kv_out, kv = self._run("lm_prefill", {"hidden": hidden_pad[None],
                                              "cos": cos_p, "sin": sin_p})
        timing["prefill"] = time.perf_counter() - t1
        last = kv_out[0, real - 1]
        kv_cache = np.zeros((self.cfg["backbone_layers"], 2, 1,
                             self.cfg["backbone_kv_heads"], self.cap, 64), np.float32)
        kv_cache[:, :, :, :, :real] = kv[:, :, :, :, :real]
        return dict(prompt=prompt, real=real, last=last, kv_cache=kv_cache, pos=real,
                    timing=timing, t0=t0, prefix=prefix, stream=stream, key=key,
                    max_frames=min(max_frames, self.cap - real))

    def synthesize_stream(self, text, prompt_wav, prompt_text, lang="Chinese", seed=1234,
                          max_frames=400, prompt_codes=None, first_frames=4, chunk_frames=8,
                          overlap=True, greedy=False, codebooks=16):
        """Yield 24 kHz PCM while the autoregressive LM is still running.

        A background thread feeds the codec decoder so its ~6.6 ms/frame cost
        overlaps with the next LM frame; the first audio packet is emitted after
        first_frames (default 4 = 320 ms) of generated codes.
        """
        import queue
        import threading

        self._prepare_tables()
        state = self._prepare(text, prompt_wav, prompt_text, lang, seed, max_frames, prompt_codes, codebooks)
        t0, timing = state["t0"], state["timing"]
        prompt, real, pos = state["prompt"], state["real"], state["pos"]
        kv_cache, last = state["kv_cache"], state["last"]
        max_frames = state["max_frames"]
        rng = np.random.default_rng(seed)
        stream = state["stream"]
        warm = stream is not None
        if stream is None:
            stream = self._CodecStream(self, keep=codebooks)
        else:
            stream.keep = codebooks
        cache_key, prefix = state["key"], state["prefix"]
        skip = [len(stream.pending) * self.hop if warm else len(prompt) * self.hop]

        def decode(codes, final=False):
            pcm = stream.push(codes, final)
            if skip[0]:
                k = min(skip[0], len(pcm))
                pcm, skip[0] = pcm[k:], skip[0] - k
            return pcm

        jobs, pcm_q, errors = queue.Queue(), queue.Queue(), []

        def codec_worker():
            while True:
                item = jobs.get()
                if item is None:
                    break
                codes, final, is_prompt = item
                try:
                    pcm = decode(codes, final)
                    if is_prompt and cache_key is not None:
                        self._prompt_cache[cache_key] = dict(
                            codes=prompt.copy(), prefix=prefix.copy(), stream=stream.clone())
                except Exception as error:  # noqa: BLE001 - surface to caller
                    errors.append(error)
                    break
                pcm_q.put(pcm)
                if final:
                    break
            pcm_q.put(None)

        worker = threading.Thread(target=codec_worker, daemon=True) if overlap else None
        if worker is not None:
            worker.start()
            if not warm:
                jobs.put((prompt, False, True))   # prompt context decoded in the background
        elif not warm:
            decode(prompt)                        # sequential warmup of the prompt context

        def collect(wait_for_audio):
            out = []
            while True:
                try:
                    item = pcm_q.get(timeout=30) if wait_for_audio else pcm_q.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    break
                if len(item):
                    out.append(item)
                    wait_for_audio = False   # first packet found: drain the rest
            if errors:
                raise errors[0]
            return out

        depth_buf = np.zeros((self.cfg["decoder_layers"], 2, 1,
                              self.cfg["decoder_kv_heads"], self.cfg["depth_slots"], 64), np.float32)
        depth_mask = np.full((1, 1, 1, self.cfg["depth_slots"] + 1), NEG, np.float32)
        depth_feed = {"local": None, "mask": depth_mask, "cos": None, "sin": None, "past": depth_buf}
        gen, eos = [], False
        align = (-len(prompt)) % self.cframes
        first_size = first_frames + align
        threshold, queued = first_size, 0
        ttft, ttfa, first_audio = None, None, True
        ar_time = 0.0
        chunks = 0

        completed = False
        try:
            frame = 0
            while frame < max_frames:
                ta = time.perf_counter()
                noise = rng.exponential(size=(16, self.cfg["audio_vocab"])).astype(np.float32)
                np.maximum(noise, self.noise_tiny, out=noise)
                c0 = self._sample(np.dot(self.head0, last), noise[0], frame > self.cfg["min_frames"], greedy=greedy)
                local = np.empty((1, 2, self.cfg["hidden"]), np.float32)
                np.dot(last, self.proj.T, out=local[0, 0])
                np.dot(self.audio_emb[c0], self.proj.T, out=local[0, 1])
                dh, dkv = self._run("depth_prefill", {"local": local})
                depth_buf[:, :, :, :, :2] = dkv
                h = dh[0]
                codes = [c0]
                for level in range(codebooks - 1):
                    ci = self._sample(np.dot(h, self.audio_head[level]), noise[level + 1], True, greedy=greedy)
                    codes.append(ci)
                    if level == codebooks - 2:
                        break
                    token_pos = level + 2
                    depth_feed["local"] = self.proj_audio[ci + (level + 1) * self.cfg["audio_vocab"]][None, None]
                    depth_mask[...] = NEG
                    depth_mask[..., :token_pos] = 0.0
                    depth_mask[..., -1] = 0.0
                    depth_feed["cos"] = self.dcos[token_pos].reshape(1, 1, 1, 64)
                    depth_feed["sin"] = self.dsin[token_pos].reshape(1, 1, 1, 64)
                    h_out, newkv = self._run("depth_step", depth_feed)
                    h = h_out[0]
                    depth_buf[:, :, :, :, token_pos] = newkv[:, :, :, :, 0]
                codes = np.array(codes, np.int64)
                frame += 1
                if ttft is None:
                    ttft = time.perf_counter() - t0
                if np.all(codes == 0):
                    eos = True
                    ar_time += time.perf_counter() - ta
                    break
                gen.append(np.clip(codes - 1, 0, self.cfg["audio_vocab"] - 2))
                if codebooks < 16:
                    # dropped codebooks contribute their average residual vector
                    next_hidden = (self.audio_emb[codes + self.offsets[:codebooks]].sum(axis=0)
                                   + self.prior[codebooks:].sum(axis=0))[None]
                else:
                    next_hidden = self.audio_emb[codes + self.offsets].sum(axis=0)[None]
                bmask = np.full((1, 1, 1, self.cap + 1), NEG, np.float32)
                bmask[..., :pos] = 0.0
                bmask[..., -1] = 0.0
                out, newkv = self._run("lm_decode", {"hidden": next_hidden[None], "mask": bmask,
                                                     "cos": self.bcos[pos].reshape(1, 1, 1, 64),
                                                     "sin": self.bsin[pos].reshape(1, 1, 1, 64),
                                                     "past": kv_cache})
                kv_cache[:, :, :, :, pos] = newkv[:, :, :, :, 0]
                last = out[0]
                pos += 1
                ar_time += time.perf_counter() - ta
                if len(gen) - queued >= threshold:
                    block = np.stack(gen[queued:])
                    queued = len(gen)
                    threshold = chunk_frames
                    if worker is not None:
                        jobs.put((block, False, False))
                        pcm = collect(first_audio)
                    else:
                        pcm = [decode(block)]
                    if pcm:
                        if ttfa is None:
                            ttfa = time.perf_counter() - t0
                        first_audio = False
                        chunks += 1
                        yield np.concatenate(pcm)
            completed = True
        finally:
            if not completed:
                jobs.put(None)
        if len(gen) > queued:
            if worker is not None:
                jobs.put((np.stack(gen[queued:]), False, False))
            else:
                pcm = decode(np.stack(gen[queued:]))
                if len(pcm):
                    if ttfa is None:
                        ttfa = time.perf_counter() - t0
                    chunks += 1
                    yield pcm
        if worker is not None:
            jobs.put((np.zeros((0, 16), np.int64), True, False))
            worker.join()
            tail = collect(False)
        else:
            tail = [decode(np.zeros((0, 16), np.int64), final=True)]
        if tail:
            if ttfa is None:
                ttfa = time.perf_counter() - t0
            chunks += 1
            yield np.concatenate(tail)
        total = time.perf_counter() - t0
        seconds = len(gen) * self.hop / self.sr
        self.generated_codes = np.stack(gen) if gen else np.zeros((0, 16), np.int64)
        timing.update(ar_loop=ar_time, codec_dec=max(0.0, total - ar_time - timing["codec_enc"] -
                                                    timing["prefill"]),
                      frames=len(gen), eos=eos, prompt_frames=len(prompt), tokens=real,
                      audio_seconds=seconds, total=total,
                      rtf=total / max(seconds, 1e-6), seed=seed, text=text, lang=lang,
                      ttft=ttft, ttfa=ttfa, chunks=chunks, streaming=True,
                      codebooks=codebooks, max_frames=max_frames)
        self.last_timing = timing

    def synthesize(self, text, prompt_wav, prompt_text, **kwargs):
        kwargs.pop("stream", None)
        chunks = list(self.synthesize_stream(text, prompt_wav, prompt_text, **kwargs))
        if not chunks:
            return np.zeros(0, np.float32)
        wav = np.concatenate(chunks)
        self.last_timing["streaming"] = False
        return wav

    def generate(self, text, prompt_wav, prompt_text, lang="Chinese", seed=1234,
                 max_frames=400, stream=False, **kwargs):
        """Single call: returns PCM (or a chunk generator when stream=True)."""
        if stream:
            return self.synthesize_stream(text, prompt_wav, prompt_text, lang=lang, seed=seed,
                                          max_frames=max_frames, **kwargs)
        return self.synthesize(text, prompt_wav, prompt_text, lang=lang, seed=seed,
                               max_frames=max_frames, **kwargs)
