# C++ 运行时（AX650 / AX_ENGINE）

纯 C++17，无 Python/numpy 依赖：WAV 读写、重采样、去静音、byte-level BPE
（`models/assets/tokenizer.bin`）、embedding / 15 个音频头 / top-k 采样、KV cache、
流式 codec 解码。codec 的 RVQ 解码（16 码本 Gather + 投影）已并入 axmodel，
host 端只传 16 个码字；host 表为 int8/fp16。

音频头 / head0 以 **int8** 常驻内存（15.7 / 1.1 MB），GEMV 用
`#pragma omp simd reduction` 向量化；`projected_audio` 用 int8 查表，
避免每帧 16 次 512×512 fp32 投影。单帧 host 头表读取从 63 MB 降到 15.7 MB。

## 编译

交叉编译（推荐，与 BSP 版本一致）：

```bash
TOOLCHAIN_ROOT=/path/to/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu \
BSP_MSP_DIR=/path/to/ax650n_bsp_sdk/msp/out \
bash cpp/build_ax650.sh
```

板端本地编译（BSP 头文件/库在 `/soc`）：

```bash
BSP_MSP_DIR=/soc CXX=g++ CC=gcc STRIP=strip bash cpp/build_ax650.sh
```

产物：`bin/m5tts_ax650`。

## 运行

```bash
./bin/m5tts_ax650 --model-dir models --assets models/assets \
    --text "你好，欢迎使用我们的语音合成模型。" \
    --prompt-wav samples/aslp_default_voice.wav \
    --prompt-text "猪携带该疾病，然后通过蚊子传染给人类。" \
    --output out.wav [--stream] [--seed 1234] [--lang Chinese] \
    [--ref-cache cache/ref]    # 同一参考音色只编码一次（TTFT 0.43 -> 0.07 s）

# 只打印文本 token ids（与 Python tokenizer 逐 id 一致）
./bin/m5tts_ax650 --dump-ids --assets models/assets \
    --prompt-text "..." --text "..."
```

`--stream` 每 320–640 ms 写一包音频，并打印 TTFT / TTFA；非流式打印分阶段耗时。
