# Models

> 本分支为 **全 NPU2** 版本：6 个 axmodel 均按 `npu_mode=NPU2` 编译；模型从 HuggingFace 的 `npu2` 分支下载。
> 标准版（LM/depth NPU1 + codec NPU3）在 `main` 分支。

The 6 axmodels and the host assets are published on HuggingFace:

    https://huggingface.co/HY-2012/M5-TTS.AXERA/tree/npu2

`assets/` 里的 host 表为 **int8 + 每行 scale**（文本/音频/head0/audio_head）和 **fp16**
（codec 编码码本），运行时加载阶段反量化；`projected_audio` 与 codec 解码 RVQ 表已不再需要。

Download the `models/` directory from that repository into this directory
(`models/` next to `bin/`, `python/`, `samples/`), or convert them yourself with
`model_convert/` (see `model_convert/README.md`).

| file | size | role |
|---|---|---|
| `lm_prefill.axmodel` | 33 MB | 256-token prompt pass |
| `lm_decode.axmodel` | 33 MB | one token + 512-slot KV |
| `depth_prefill.axmodel` | 21 MB | first 2 local tokens |
| `depth_step.axmodel` | 21 MB | one codebook position |
| `codec_encoder.axmodel` | 44 MB | 4.8 s reference -> [512,60] features |
| `codec_decoder.axmodel` | 133 MB | 8 frames of codes -> 15360 samples（RVQ 解码在图内） |
| `assets/` | 159 MB | int8/fp16 embeddings、heads、codec 编码码本、tokenizer |
