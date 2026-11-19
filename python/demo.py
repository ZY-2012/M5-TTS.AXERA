#!/usr/bin/env python3
"""M5-TTS (AX650) command line demo.

    python3 demo.py --model-dir models --assets assets \
        --text "你好，欢迎使用我们的语音合成模型。" \
        --prompt-wav samples/aslp_default_voice.wav \
        --prompt-text "猪携带该疾病，然后通过蚊子传染给人类。" --output out.wav

    --stream writes audio packets while the LM is still generating (TTFA).
"""
import argparse
import json
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m5_ax import M5AX, write_wav  # noqa: E402


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("M5_MODEL_DIR", "models"))
    local_assets = os.path.join(here, "assets")
    default_assets = local_assets if os.path.isdir(local_assets) else os.path.join("models", "assets")
    ap.add_argument("--assets", default=os.environ.get("M5_ASSETS", default_assets))
    ap.add_argument("--engine", default="ax", choices=["ax", "onnx"])
    ap.add_argument("--text", default="你好，欢迎使用我们的语音合成模型。")
    ap.add_argument("--prompt-wav", default=os.path.join(here, "samples", "aslp_default_voice.wav"))
    ap.add_argument("--prompt-text", default="猪携带该疾病，然后通过蚊子传染给人类。")
    ap.add_argument("--lang", default="Chinese")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--stream", action="store_true",
                    help="write packets while generating (lower TTFA)")
    ap.add_argument("--first-frames", type=int, default=4, help="codes in the first packet")
    ap.add_argument("--chunk-frames", type=int, default=8, help="codes per packet afterwards")
    ap.add_argument("--no-overlap", action="store_true",
                    help="decode the codec sequentially (benchmark A/B)")
    ap.add_argument("--repeat", type=int, default=1, help="repeat runs (warm reference cache)")
    ap.add_argument("--output", default="out.wav")
    a = ap.parse_args()
    tts = M5AX(a.model_dir, a.assets, engine=a.engine)
    for _ in range(max(1, a.repeat) - 1):
        tts.generate(a.text, a.prompt_wav, a.prompt_text, lang=a.lang, seed=a.seed,
                     max_frames=min(a.max_frames, 4))
    started = time.perf_counter()
    if a.stream:
        with wave.open(a.output, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(tts.sr)
            for pcm in tts.synthesize_stream(a.text, a.prompt_wav, a.prompt_text, lang=a.lang,
                                             seed=a.seed, max_frames=a.max_frames,
                                             first_frames=a.first_frames,
                                             chunk_frames=a.chunk_frames,
                                             overlap=not a.no_overlap):
                handle.writeframes((np.clip(pcm, -1, 1) * 32767.0).astype("<i2").tobytes())
        tm = tts.last_timing
        print(f"stream: {tm['chunks']} packets, TTFT {tm['ttft']:.2f}s, "
              f"TTFA {tm['ttfa']:.2f}s, audio {tm['audio_seconds']:.2f}s in "
              f"{tm['total']:.2f}s (RTF {tm['rtf']:.2f})")
    else:
        wav = tts.generate(a.text, a.prompt_wav, a.prompt_text, lang=a.lang,
                           seed=a.seed, max_frames=a.max_frames,
                           overlap=not a.no_overlap)
        write_wav(a.output, wav, tts.sr)
        tm = tts.last_timing
        if tm.get("ttfa") is not None:
            print(f"TTFT {tm['ttft']:.2f}s, TTFA {tm['ttfa']:.2f}s")
    tm = dict(tts.last_timing)
    tm["wall"] = time.perf_counter() - started
    print(json.dumps(tm, indent=2, ensure_ascii=False))
    print("saved", a.output)


if __name__ == "__main__":
    main()
