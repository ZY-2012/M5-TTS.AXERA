"""Explicit static-shape graphs for M5-TTS (backbone / MTP depth / Qwen3 codec).

Only the NPU-friendly parts live in the graphs:

  lm_prefill   hidden[S,512]                    -> hidden[S,512] + KV[8,2,1,2,S,64]
  lm_decode    1 token + fixed KV + additive mask -> hidden[1,512] + new KV
  depth_prefill projected[2,512]                -> hidden[1,512] + KV[5,2,1,2,2,64]
  depth_step   1 token + fixed KV + mask        -> hidden[1,512] + new KV
  codec_enc    wav[1,115200]                    -> codes[1,16,60]
  codec_dec    latent[1,512,N] + states         -> wav[1,N*1920] + states

Sampling (top-k / softmax / argmax), the 15 codebook heads, the projection and
all embedding tables stay on the host in fp32, mirroring the CuteTTS split.
"""

from __future__ import annotations

import torch
from torch import nn

NEG = -30.0


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class LlamaCore(nn.Module):
    """Llama3-style stack with explicit tensor KV state (no HF cache objects)."""

    def __init__(self, decoder):
        super().__init__()
        self.layers = decoder.layers
        self.norm = decoder.norm
        cfg = decoder.config
        self.heads = cfg.num_attention_heads
        self.kv_heads = cfg.num_key_value_heads
        self.dim = cfg.hidden_size // self.heads
        self.scale = self.dim ** -0.5
        self.repeats = self.heads // self.kv_heads

    def _layer(self, layer, hidden, cos, sin, mask, past_k, past_v):
        attn = layer.self_attn
        x = layer.input_layernorm(hidden)
        q = attn.q_proj(x)
        k = attn.k_proj(x)
        v = attn.v_proj(x)
        t = q.shape[1]
        q = q.reshape(1, t, self.heads, self.dim).transpose(1, 2)
        k = k.reshape(1, t, self.kv_heads, self.dim).transpose(1, 2)
        v = v.reshape(1, t, self.kv_heads, self.dim).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        if past_k is not None:
            k_all = torch.cat((past_k, k), dim=-2)
            v_all = torch.cat((past_v, v), dim=-2)
        else:
            k_all, v_all = k, v
        keys = k_all[:, :, None].expand(-1, -1, self.repeats, -1, -1).reshape(
            1, self.heads, -1, self.dim)
        vals = v_all[:, :, None].expand(-1, -1, self.repeats, -1, -1).reshape(
            1, self.heads, -1, self.dim)
        scores = (q @ keys.transpose(-1, -2)) * self.scale + mask
        out = scores.softmax(-1) @ vals
        out = out.transpose(1, 2).reshape(1, t, self.heads * self.dim)
        hidden = hidden + attn.o_proj(out)
        x = layer.post_attention_layernorm(hidden)
        hidden = hidden + layer.mlp(x)
        return hidden, k, v

    def stack(self, pairs):
        return torch.stack([torch.stack(p) for p in pairs])


class Prefill(nn.Module):
    """Full prompt pass; KV for every position is returned."""

    def __init__(self, decoder, seq):
        super().__init__()
        self.core = LlamaCore(decoder)
        self.seq = seq
        causal = torch.tril(torch.ones(seq, seq)) * 0.0
        causal = causal + torch.triu(torch.full((seq, seq), NEG), diagonal=1)
        self.register_buffer("mask", causal.view(1, 1, seq, seq))
        positions = torch.arange(seq).float()
        inv = decoder.position_embedding.inv_freq.float()
        freqs = positions[:, None] * inv[None, :]
        freqs = torch.cat((freqs, freqs), dim=-1)
        scale = float(decoder.position_embedding.attention_scaling)
        self.register_buffer("cos", (freqs.cos() * scale).view(1, 1, seq, -1))
        self.register_buffer("sin", (freqs.sin() * scale).view(1, 1, seq, -1))

    def forward(self, hidden, cos, sin):
        pairs = []
        for layer in self.core.layers:
            hidden, k, v = self.core._layer(layer, hidden, cos, sin, self.mask, None, None)
            pairs.append((k, v))
        return self.core.norm(hidden), self.core.stack(pairs)


class Decode(nn.Module):
    """One token against a fixed-capacity KV buffer; returns only the new KV."""

    def __init__(self, decoder, cap):
        super().__init__()
        self.core = LlamaCore(decoder)
        self.cap = cap

    def forward(self, hidden, cos, sin, mask, past):
        pairs = []
        for index, layer in enumerate(self.core.layers):
            hidden, k, v = self.core._layer(layer, hidden, cos, sin, mask,
                                            past[index, 0], past[index, 1])
            pairs.append((k, v))
        return self.core.norm(hidden)[:, 0], self.core.stack(pairs)


class DepthPrefill(nn.Module):
    """Two local tokens (backbone hidden + codebook0 embedding projection)."""

    def __init__(self, decoder):
        super().__init__()
        self.core = LlamaCore(decoder)
        positions = torch.arange(2).float()
        inv = decoder.position_embedding.inv_freq.float()
        freqs = positions[:, None] * inv[None, :]
        freqs = torch.cat((freqs, freqs), dim=-1)
        scale = float(decoder.position_embedding.attention_scaling)
        self.register_buffer("cos", (freqs.cos() * scale).view(1, 1, 2, -1))
        self.register_buffer("sin", (freqs.sin() * scale).view(1, 1, 2, -1))
        causal = torch.tril(torch.ones(2, 2)) * 0.0
        causal = causal + torch.triu(torch.full((2, 2), NEG), diagonal=1)
        self.register_buffer("mask", causal.view(1, 1, 2, 2))

    def forward(self, local):
        pairs = []
        hidden = local
        for layer in self.core.layers:
            hidden, k, v = self.core._layer(layer, hidden, self.cos, self.sin,
                                            self.mask, None, None)
            pairs.append((k, v))
        return self.core.norm(hidden)[:, 1], self.core.stack(pairs)


class DepthStep(nn.Module):
    """One codebook position against a 16-slot local KV buffer."""

    def __init__(self, decoder):
        super().__init__()
        self.core = LlamaCore(decoder)

    def forward(self, local, cos, sin, mask, past):
        pairs = []
        hidden = local
        for index, layer in enumerate(self.core.layers):
            hidden, k, v = self.core._layer(layer, hidden, cos, sin, mask,
                                            past[index, 0], past[index, 1])
            pairs.append((k, v))
        return self.core.norm(hidden)[:, 0], self.core.stack(pairs)


class CodecEncoder(nn.Module):
    """Causal codec encoder, fixed 4.8 s window (60 frames @ 12.5 Hz)."""

    def __init__(self, codec, samples=115200):
        super().__init__()
        self.encoder = codec.encoder
        self.samples = samples

    def forward(self, audio):
        x = self.encoder.encoder(audio[:, None], padding_cache=None)
        x = self.encoder.encoder_transformer(x.transpose(1, 2))[0].transpose(1, 2)
        return self.encoder.downsample(x, padding_cache=None)


class CodecDecoder(nn.Module):
    """Causal streaming decoder taking pre-summed latents (host does the RVQ sum).

    A copy of the upstream cached decoder with RoPE cos/sin supplied by the host
    so no trigonometric op ends up inside the quantized graph.
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        config = decoder.pre_transformer.config
        self.layers = config.num_hidden_layers
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.dim = config.head_dim
        self.window = config.sliding_window
        if not self.window:
            raise ValueError("This export requires bounded sliding attention")
        self.specs = []
        for name, module in decoder.named_modules():
            kind = type(module).__name__
            if kind == "Qwen3TTSTokenizerV2CausalConvNet":
                if module.stride != 1:
                    raise ValueError(f"Unsupported strided causal convolution: {name}")
                if module.padding:
                    self.specs.append(dict(name=name, shape=[1, module.conv.in_channels, module.padding]))
            elif kind == "Qwen3TTSTokenizerV2CausalTransConvNet" and module.right_pad:
                self.specs.append(dict(name=name, shape=[1, module.conv.out_channels, module.right_pad]))

    def empty_states(self):
        return [torch.zeros(spec["shape"]) for spec in self.specs]

    def _convolution(self, module, hidden, states, updated):
        kind = type(module).__name__
        if kind == "Qwen3TTSTokenizerV2CausalConvNet":
            if module.padding:
                hidden = torch.cat((next(states), hidden), dim=-1)
                updated.append(hidden[..., -module.padding:])
            return module.conv(hidden)
        output = module.conv(hidden)
        padding = module.right_pad
        if not padding:
            return output
        previous = next(states)
        bias = module.conv.bias[None, :, None] if module.conv.bias is not None else 0.0
        updated.append(output[..., -padding:] - bias)
        overlap = output[..., :padding] + previous
        return torch.cat((overlap, output[..., padding:-padding]), dim=-1)

    def _block(self, module, hidden, states, updated):
        kind = type(module).__name__
        if kind in ("Qwen3TTSTokenizerV2CausalConvNet", "Qwen3TTSTokenizerV2CausalTransConvNet"):
            return self._convolution(module, hidden, states, updated)
        if kind == "Qwen3TTSTokenizerV2ConvNeXtBlock":
            residual = hidden
            hidden = self._block(module.dwconv, hidden, states, updated).transpose(1, 2)
            hidden = module.pwconv2(module.act(module.pwconv1(module.norm(hidden))))
            return residual + (module.gamma * hidden).transpose(1, 2)
        if kind == "Qwen3TTSTokenizerV2DecoderDecoderResidualUnit":
            residual = hidden
            hidden = self._block(module.conv1, module.act1(hidden), states, updated)
            return residual + self._block(module.conv2, module.act2(hidden), states, updated)
        if kind == "Qwen3TTSTokenizerV2DecoderDecoderBlock":
            for block in module.block:
                hidden = self._block(block, hidden, states, updated)
            return hidden
        return module(hidden)

    def _transformer(self, hidden, past, cos, sin, mask):
        module = self.decoder.pre_transformer
        hidden = module.input_proj(hidden)
        if cos.ndim == 3:
            cos, sin = cos[:, None], sin[:, None]
        present = []
        for index, layer in enumerate(module.layers):
            attn = layer.self_attn
            x = layer.input_layernorm(hidden)
            q = attn.q_norm(attn.q_proj(x).reshape(1, -1, self.heads, self.dim)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(x).reshape(1, -1, self.kv_heads, self.dim)).transpose(1, 2)
            v = attn.v_proj(x).reshape(1, -1, self.kv_heads, self.dim).transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            k_all = torch.cat((past[index, 0], k), dim=-2)
            v_all = torch.cat((past[index, 1], v), dim=-2)
            present.append(torch.stack((k_all[..., -(self.window - 1):, :],
                                        v_all[..., -(self.window - 1):, :])))
            repeats = self.heads // self.kv_heads
            keys = k_all[:, :, None].expand(-1, -1, repeats, -1, -1).reshape(
                1, self.heads, -1, self.dim)
            values = v_all[:, :, None].expand(-1, -1, repeats, -1, -1).reshape(
                1, self.heads, -1, self.dim)
            weights = ((q @ keys.transpose(-1, -2)) * attn.scaling + mask).softmax(-1)
            out = (weights @ values).transpose(1, 2).reshape(1, -1, self.heads * self.dim)
            hidden = hidden + layer.self_attn_layer_scale(attn.o_proj(out))
            hidden = hidden + layer.mlp_layer_scale(layer.mlp(layer.post_attention_layernorm(hidden)))
        return module.output_proj(module.norm(hidden)), torch.stack(present)

    def forward(self, latent, past, cos, sin, mask, *convolution_states):
        states, updated = iter(convolution_states), []
        hidden = self._block(self.decoder.pre_conv, latent, states, updated).transpose(1, 2)
        hidden, present = self._transformer(hidden, past, cos, sin, mask)
        hidden = hidden.transpose(1, 2)
        for blocks in self.decoder.upsample:
            for block in blocks:
                hidden = self._block(block, hidden, states, updated)
        for block in self.decoder.decoder:
            hidden = self._block(block, hidden, states, updated)
        return (hidden.clamp(-1, 1), present, *updated)


def causal_extra_padding(module, hidden):
    return (-hidden.shape[-1]) % module.stride


def export_graph(module, inputs, path, names, outputs):
    import onnx
    torch.onnx.export(module.eval(), inputs, str(path), input_names=names,
                      output_names=outputs, opset_version=17, dynamo=False,
                      do_constant_folding=True)
    onnx.checker.check_model(str(path))
    print(f"exported {path.name}", flush=True)


class StaticMimiAttention(nn.Module):
    """Mimi attention with integer reshapes so Pulsar2 sees constant shapes."""

    def __init__(self, attn, seq):
        super().__init__()
        self.attn = attn
        self.seq = seq

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        a = self.attn
        t, b = self.seq, 1
        q = a.q_proj(hidden_states).reshape(b, t, a.num_heads, a.head_dim).transpose(1, 2)
        k = a.k_proj(hidden_states).reshape(b, t, a.num_key_value_heads, a.head_dim).transpose(1, 2)
        v = a.v_proj(hidden_states).reshape(b, t, a.num_key_value_heads, a.head_dim).transpose(1, 2)
        cos, sin = a.rotary_emb(v, position_ids)
        q = q * cos[:, None] + rotate_half(q) * sin[:, None]
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
        rep = a.num_key_value_groups
        keys = k[:, :, None].expand(b, a.num_key_value_heads, rep, t, a.head_dim).reshape(
            b, a.num_heads, t, a.head_dim)
        vals = v[:, :, None].expand(b, a.num_key_value_heads, rep, t, a.head_dim).reshape(
            b, a.num_heads, t, a.head_dim)
        weights = (q @ keys.transpose(2, 3)) * a.scaling
        if attention_mask is not None:
            weights = weights + attention_mask[:, :, :, :t]
        weights = weights.softmax(-1)
        out = (weights @ vals).transpose(1, 2).reshape(b, t, a.num_heads * a.head_dim)
        return a.o_proj(out), None


class EncoderTransformer(nn.Module):
    """Causal Mimi encoder transformer with a baked 0/-30 additive mask."""

    def __init__(self, original, seq):
        super().__init__()
        self.original = original
        self.seq = seq
        for layer in original.layers:
            layer.self_attn = StaticMimiAttention(layer.self_attn, seq)

    def forward(self, hidden_states=None, inputs_embeds=None, **kwargs):
        x = inputs_embeds if inputs_embeds is not None else hidden_states
        positions = torch.arange(self.seq)
        distance = positions[:, None] - positions[None, :]
        allowed = distance >= 0
        window = getattr(self.original.config, "sliding_window", None)
        if window:
            allowed = allowed & (distance < window)
        mask = torch.where(allowed, 0.0, NEG)[None, None]
        for layer in self.original.layers:
            out = layer(x, attention_mask=mask, position_ids=positions[None], use_cache=False)
            x = out[0] if isinstance(out, tuple) else out
        return (x,)


class StaticMimiConv1d(nn.Module):
    """Mimi causal conv with a constant pad tuple (Pulsar2 cannot execute the
    dynamic `_get_extra_padding_for_conv1d` shape arithmetic)."""

    def __init__(self, original, length):
        super().__init__()
        import math
        self.conv = original.conv
        kernel = original.conv.kernel_size[0]
        stride = original.conv.stride[0]
        dilation = original.conv.dilation[0]
        effective = (kernel - 1) * dilation + 1
        padding_total = effective - stride
        n_frames = math.ceil((length - effective + padding_total) / stride + 1) - 1
        ideal = n_frames * stride + effective - padding_total
        self.padding_left = padding_total if original.causal else padding_total // 2
        self.padding_right = (0 if original.causal else padding_total - padding_total // 2) + int(ideal - length)
        self.mode = original.pad_mode

    def forward(self, hidden_states, padding_cache=None):
        # Pulsar2 has no edge-mode Pad; replicate via slice+expand
        if self.mode == "constant":
            return self.conv(torch.nn.functional.pad(hidden_states, (int(self.padding_left), int(self.padding_right))))
        if self.padding_left:
            left = hidden_states[..., :1].expand(-1, -1, int(self.padding_left))
            hidden_states = torch.cat((left, hidden_states), dim=-1)
        if self.padding_right:
            right = hidden_states[..., -1:].expand(-1, -1, int(self.padding_right))
            hidden_states = torch.cat((hidden_states, right), dim=-1)
        return self.conv(hidden_states)
