"""Graph variants that pull the remaining host tables into the axmodels.

Design notes (see ../docs/feasibility.md for the Pulsar2 evidence):

* Pulsar2 gathers tables with **int16** indices -> every table here is <= 32767 rows.
* Multi-gather graphs are fragile; each graph uses at most **2** gathers with
  host-precomputed flat indices (``level * rows + code``).
* Dynamic head selection via Gather fails -> the 15 heads are summed with a
  host-provided one-hot, which keeps the logits output at 2051 values.

Only the four graphs below differ from ``../../M5-TTS-distill/export/m5_graphs.py``;
``lm_prefill``, ``lm_decode`` and ``codec_encoder`` are reused unchanged.
"""

from __future__ import annotations

import torch
from torch import nn

try:  # only needed for the codec wrapper; keeps this module importable standalone
    from m5_graphs import CodecDecoder as _CodecDecoder
except Exception:  # pragma: no cover
    _CodecDecoder = nn.Module

# ---------------------------------------------------------------------- tables


class AudioEmbedFrame(nn.Module):
    """16 codes -> (summed LM embedding, projected embedding per level).

    Tables (fp16 in the axmodel)::

        t0 = audio_emb[0:8].reshape(8*2051, 512)   # levels 0..7
        t1 = audio_emb[8:16].reshape(8*2051, 512)  # levels 8..15
        proj = projection                            # [512, 512]

    Inputs:  i0, i1  int32 [1, 8]   flat = level_in_table * 2051 + code
    Outputs: emb     [1, 512]       sum of all 16 embeddings (LM input)
             local   [1, 16, 512]   proj @ embedding per level (depth decoder)
    """

    def __init__(self, audio_emb, proj):
        super().__init__()
        a = torch.as_tensor(audio_emb, dtype=torch.float32).reshape(16, 2051, 512)
        self.register_buffer("t0", a[:8].reshape(8 * 2051, 512))
        self.register_buffer("t1", a[8:].reshape(8 * 2051, 512))
        self.register_buffer("proj", torch.as_tensor(proj, dtype=torch.float32))

    def forward(self, i0, i1):
        e0 = self.t0[i0]                      # [1, 8, 512]
        e1 = self.t1[i1]                      # [1, 8, 512]
        emb16 = torch.cat((e0, e1), dim=1)    # [1, 16, 512]
        return emb16.sum(dim=1), emb16 @ self.proj.t()


class DepthPrefillHead(nn.Module):
    """depth_prefill + codebook-1 head:  local[1,2,512] -> h, logits, KV.

    ``head`` is ``audio_head[0]`` (shape [512, 2051]); the LM ``head0`` stays on
    the host because codebook 0 is sampled from the backbone hidden *before* the
    depth prefill runs.
    """

    def __init__(self, core: nn.Module, head):
        super().__init__()
        self.core = core
        self.register_buffer("head", torch.as_tensor(head, dtype=torch.float32))

    def forward(self, local):
        h, kv = self.core(local)
        return h, h @ self.head, kv


class DepthStepHeads(nn.Module):
    """depth_step + heads 1..14 selected by a 14-way one-hot (head 0 is in depth_prefill).

    Inputs:  local [1,1,512], onehot [14], cos, sin, mask, past
    Outputs: h [1,512], logits [1,2051], KV
    """

    def __init__(self, core: nn.Module, audio_head):
        super().__init__()
        self.core = core
        w = torch.as_tensor(audio_head, dtype=torch.float32)   # [15,512,2051]
        assert w.shape == (15, 512, 2051), w.shape
        w = w[1:]                                              # heads 1..14
        self.register_buffer("w", w.transpose(1, 2).reshape(14 * 2051, 512).contiguous())

    def forward(self, local, onehot, cos, sin, mask, past):
        h, kv = self.core(local, cos, sin, mask, past)
        logits_all = (h @ self.w.t()).reshape(1, 14, 2051)
        logits = (logits_all * onehot.reshape(1, 14, 1)).sum(dim=1)
        return h, logits, kv


class CodecDecoderRVQ(_CodecDecoder):
    """codec decoder taking codes instead of a pre-summed latent.

    Subclasses the streaming decoder so the conv-state inputs keep their original
    names (a wrapper module re-traces them as ``conv_*.1`` and ORT rejects the feed).

    Tables:
        sem  = book_sem[0]            [2048, 256]
        rest = book_rest.reshape(15*2048, 256)

    Inputs:  i0 [1, T]  (semantic code), i1 [1, 15*T] (level 1..15 flat indices),
             past, cos, sin, mask, states
    Outputs: wav, present, states

    Latent: z = (proj_sem @ sem + b_sem) + (proj_rest @ sum(rest) + b_rest)
    """

    def __init__(self, decoder: nn.Module, book_sem, book_rest, ps_w, ps_b, pr_w, pr_b):
        super().__init__(decoder)
        sem = torch.as_tensor(book_sem, dtype=torch.float32)
        rest = torch.as_tensor(book_rest, dtype=torch.float32)
        assert sem.shape[-2:] == (2048, 256), sem.shape
        assert rest.shape == (15, 2048, 256), rest.shape
        self.register_buffer("sem", sem.reshape(2048, 256))
        self.register_buffer("rest", rest.reshape(15 * 2048, 256))
        self.register_buffer("ps_w", torch.as_tensor(ps_w, dtype=torch.float32))
        self.register_buffer("ps_b", torch.as_tensor(ps_b, dtype=torch.float32))
        self.register_buffer("pr_w", torch.as_tensor(pr_w, dtype=torch.float32))
        self.register_buffer("pr_b", torch.as_tensor(pr_b, dtype=torch.float32))

    def forward(self, i0, i1, past, cos, sin, mask, *states):
        sem = self.sem[i0]                          # [1, T, 256]
        rest = self.rest[i1].reshape(1, -1, 15, 256).sum(dim=2)   # [1, T, 256]
        z = (sem @ self.ps_w.t() + self.ps_b) + (rest @ self.pr_w.t() + self.pr_b)
        return super().forward(z.transpose(1, 2).contiguous(), past, cos, sin, mask, *states)


# ------------------------------------------------------------------ host index


def flat_indices(codes, first_level: int = 0, rows: int = 2051):
    """codes [T, 16] -> (i0 [T*8], i1 [T*8]) for the two 8-level tables.

    ``first_level`` is 0 for the 16-codebook audio table (2051 rows per level).
    """
    codes = codes.astype("int32")
    levels = codes.shape[1]
    assert levels == 16, codes.shape
    off = (np.arange(8, dtype="int32") * rows)[None, :] + first_level * rows
    i0 = (codes[:, :8] + off).reshape(-1)
    i1 = (codes[:, 8:] + off).reshape(-1)
    return i0, i1


def codec_indices(codes):
    """codes [T, 16] -> (sem [T], rest [T*15]) for the codec quantizer tables."""
    codes = codes.astype("int32")
    assert codes.shape[1] == 16, codes.shape
    i0 = codes[:, 0]
    off = (np.arange(15, dtype="int32") * 2048)[None, :]
    i1 = (codes[:, 1:] + off).reshape(-1)
    return i0, i1


import numpy as np  # noqa: E402  (kept last so the module imports torch-free helpers)
