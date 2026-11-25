# M5-TTS → AX650 转换流程

本目录包含从 HuggingFace 权重到 6 个 axmodel 的完整可复现脚本。

## 0. 准备

```bash
pip install torch transformers==4.57.3 onnx onnxruntime soundfile scipy tokenizers
# Pulsar2 8.0（Magnetar）: /root/.cache/magnetar/pulsar2/8.0/bin/pulsar2
```

上游资产放到 `origin/`：

```
origin/
  M5-TTS/                         # https://github.com/ASLP-lab/M5-TTS （可用 M5_SRC 指定）
  M5-TTS-50M-VoiceClone/          # model.safetensors / config.json / conditioning.json / tokenizer/
  M5-TTS-Audio-Tokenizer/         # Qwen3-TTS 12 Hz codec（config/model.safetensors）
```

> `M5_LM_BUNDLE` / `M5_CODEC_BUNDLE` / `M5_SRC` 环境变量可覆盖路径；
> transformers 必须 4.57.x（5.x 的 `add_special_tokens` 与上游代码不兼容，可用
> `pip install --target .pylib transformers==4.57.3 huggingface-hub==0.34.4` 隔离）。

## 1. 导出 ONNX（6 个子图）

```bash
python model_convert/export_onnx.py        # lm_prefill/lm_decode/depth_prefill/depth_step/codec_encoder/codec_decoder
python model_convert/fix_reshape.py        # 把动态 Reshape/Expand shape 折叠成常量
python model_convert/export_cpp_assets.py  # tokenizer.bin / codec 表 / m5_config.h
```

每个图都用 onnxruntime 与 PyTorch 逐张量对比（cosine 全部 1.0）；
codec encoder 与原始模型的码本输出逐 id 一致（100%）。

## 2. 校准集

```bash
M5_CALIB_WAVS=/path/to/wavs python model_convert/make_calib.py
```

脚本跑完整 FP32 参考 pipeline，记录每个子图的**真实输入**：prompt embedding、
真实 KV cache、depth 的局部 KV、codec 的卷积状态与 latent、不同位置的 mask。
默认 12 条中英语音；`M5_CALIB_WAVS` 可指定自己的 wav 目录。

## 3. 编译

```bash
python model_convert/prepare_compile.py              # 只生成配置
python model_convert/prepare_compile.py --run        # 调用 pulsar2 build
python model_convert/prepare_compile.py --models depth_step --run
```

配置要点（与 CuteTTS 一致）：

- `target_hardware=AX650`，transformer 用 `NPU1`，codec 用 `NPU3`
- `layer_configs`: 激活 U16 / 权重 S8 / 输出 U16，`highest_mix_precision=false`
- `enable_smooth_quant=true`、`calibration_method=MinMax`
- `compiler.check=0`（全部上板验证）

产物在 `compile/*.axmodel`，日志 `compile/logs/`。

## 4. all-NPU 量化（int8 host 表 + 图内 RVQ）

在上面的 6 个 FP32 图基础上，把 codec 解码的 RVQ 搬进图、host 表压成 int8/fp16：

```bash
python model_convert/allnpu/export_allnpu.py          # 新 codec_decoder（codes -> wav）
python model_convert/quantize_assets.py               # int8 host 表 + fp16 编码码本
python model_convert/allnpu/prepare_compile.py --run  # Pulsar2 编译
```

- codec decoder 输入改为 `i0[1,8]`（codebook 0）+ `i1[1,120]`（level 1..15，8 帧），
  图内 2 个 FP16 码本 Gather + 投影，替代 host 的 RVQ 求和。
- 文本/音频/head0/audio_head 存 int8（每行 scale），`projected_audio` 删除，
  包体积 899 MB → 426 MB。
- 音频头保持 host 计算：S8 头 + U16 logits 在 greedy 下不稳定（详见 docs/feasibility.md）。

## 6. NPU1 变体

把 6 个图的 `npu_mode` 全部改为 `NPU1` 后重编（量化配置不变）：

- LM/depth 的 4 个 config 本来就是 NPU1，可直接复用；
- 只需重编 `codec_encoder` / `codec_decoder`（`allnpu/pulsar2_config_codec_decoder.json` + 基础 codec config）。

每个模型固定占 1 个 NPU 核，最多可 3 个模型并行。C++ RTF 0.59–0.78；
codec 因 NPU1 tiling 与 main 存在波形级数值差异，ASR 一致。
