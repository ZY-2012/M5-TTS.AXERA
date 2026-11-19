# C++ 运行时（AX650 / AX_ENGINE）

纯 C++17，无 Python/numpy 依赖：WAV 读写、重采样、去静音、byte-level BPE
（`models/assets/tokenizer.bin`）、embedding / RVQ / 15 个音频头、top-k 采样、
KV cache、流式 codec 解码。

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
    [--ref-cache cache/ref]    # 同一参考音色只编码一次（TTFT 0.30 -> 0.07 s）

# 实验性：只解码前 N 个码本（默认 16；12/8 会降音质，且 LM 可能复读，见性能文档）
./bin/m5tts_ax650 --codebooks 12 ...

# 只打印文本 token ids（与 Python tokenizer 逐 id 一致）
./bin/m5tts_ax650 --dump-ids --assets models/assets \
    --prompt-text "..." --text "..."
```

`--stream` 每 320–640 ms 写一包音频，并打印 TTFT / TTFA；非流式打印分阶段耗时。
