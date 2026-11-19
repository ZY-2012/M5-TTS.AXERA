#!/usr/bin/env python3
"""Build the real-speech calibration set and per-graph eval references.

Runs the FP32 ONNX reference pipeline over several real utterances and records
every NPU graph's real inputs (prompt embeddings, KV caches, depth-step states,
codec latents / conv states, real reference audio). The same records are replayed
through the FP32 graphs to produce the board-side accuracy baselines.

    python export/make_calib.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "export"))
import common as C
from common import CODEC_FRAMES, CODEC_SAMPLES, SEQ
from ref_pipeline import RefPipeline

DATA = ROOT.parents[2] / "TTS_Data" / "WenetSpeech4TTS" / "Premium" / "WenetSpeech4TTS_Premium_2" / "wavs"
CALIB = ROOT / "calib" / "data"
EVAL = ROOT / "export" / "eval_io"

# (reference wav, prompt transcript, target text, language, seed)
UTTERANCES = [
    (ROOT.parents[1] / "M5-TTS" / "assets" / "aslp_default_voice.wav",
     "猪携带该疾病，然后通过蚊子传染给人类。", "你好，欢迎使用我们的语音合成模型。", "Chinese", 11),
    (DATA / "X0000002644_90987046_S00063-S00064.wav",
     "今天的天气非常好。", "我们下午一起去公园散步吧。", "Chinese", 12),
    (DATA / "X0000013347_70833291_S00092-S00093.wav",
     "这是一个测试。", "语音合成系统正在运行。", "Chinese", 13),
    (DATA / "X0000015098_142383443_S00216-S00218.wav",
     "请注意安全。", "列车即将进站，请站在黄线以内。", "Chinese", 14),
    (DATA / "X0000014320_217970071_S00192-S00194.wav",
     "会议马上开始。", "请大家把手机调成静音模式。", "Chinese", 15),
    (DATA / "Y0000001196_2BUQ8U0JJW8_S01622-S01623.wav",
     "他昨天来过了。", "这个问题我们需要再讨论一下。", "Chinese", 16),
    (DATA / "Y0000010367_QhlT38rLrJM_S00419-S00422.wav",
     "谢谢你的帮助。", "以后有需要随时联系我。", "Chinese", 17),
    (DATA / "Y0000017607_jHBrQOvcYNE_S00585-S00597.wav",
     "小朋友们都很开心。", "周末我们打算去郊外野餐。", "Chinese", 18),
    (DATA / "Y0000006112_FMuf3yByHH8_S00022.wav",
     "The weather is nice today.", "Let us go for a walk in the park.", "English", 19),
    (DATA / "Y0000010410_Qq_VIYuOtOE_S00000-S00001.wav",
     "This is a short example.", "The speech model is working well.", "English", 20),
    (DATA / "Y0000019634_ocXVLeC1UVI_S00489-S00490.wav",
     "I will call you later.", "Please send me the report by Friday.", "English", 21),
    (DATA / "Y0000012855_X8wdI9hhUOs_S00209-S00214.wav",
     "We are ready to begin.", "The meeting starts in five minutes.", "English", 22),
]

LIMITS = {"lm_prefill": 14, "lm_decode": 48, "depth_prefill": 64, "depth_step": 64,
          "codec_encoder": 14, "codec_decoder": 32}


def save_inputs(name, records):
    records = records[:LIMITS[name]]
    base = CALIB / name
    tensors = sorted(records[0].keys())
    for tensor in tensors:
        folder = base / tensor
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.npy"):
            old.unlink()
        for index, row in enumerate(records):
            np.save(folder / f"{index:05d}.npy", np.ascontiguousarray(row[tensor], np.float32))
    print(f"calib {name:14s} {len(records):3d} samples  {tensors}")


def save_eval(pipe, name, records):
    """Replay 2 real samples through the FP32 graph and keep inputs + outputs."""
    records = records[:2]
    outs = []
    inputs = {}
    sess = pipe.sess[name]
    wanted = [i.name for i in sess.get_inputs()]
    for row in records:
        feed = {k: np.ascontiguousarray(row[k], np.float32) for k in wanted}
        outs.append(sess.run(None, feed))
    for k in wanted:
        inputs[k] = np.stack([np.asarray(row[k], np.float32) for row in records])
    folder = EVAL / name
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.npy"):
        old.unlink()
    for k, v in inputs.items():
        np.save(folder / f"in_{k}.npy", v)
    out_names = [o.name for o in sess.get_outputs()]
    for index, name_out in enumerate(out_names):
        stacked = np.stack([np.asarray(run[index], np.float32) for run in outs])
        np.save(folder / f"out_{name_out}.npy", stacked)
    print(f"eval  {name:14s} inputs {list(inputs)} outputs {out_names}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=len(UTTERANCES))
    args = ap.parse_args()
    CALIB.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)
    pipe = RefPipeline()
    record = {}
    summary = []
    for wav, prompt_text, text, lang, seed in UTTERANCES[:args.limit]:
        name = Path(wav).stem
        print(f"== {name} ({lang})", flush=True)
        audio = C.read_reference(str(wav))
        frames = pipe.generate(text, audio, prompt_text, lang=lang, seed=seed,
                               max_frames=120, record=record, record_every=8)
        summary.append(dict(wav=str(wav), text=text, prompt_text=prompt_text, lang=lang,
                            seed=seed, frames=int(pipe.stats["frames"]),
                            seconds=float(len(frames) / pipe.sr),
                            codes=pipe.generated_codes.tolist()))
        print(f"   {pipe.stats}", flush=True)
    for name in LIMITS:
        if name in record:
            save_inputs(name, record[name])
            save_eval(pipe, name, record[name])
    (ROOT / "calib" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    # keep one complete FP32 pipeline reference (codes + wav) for the board run
    np.save(ROOT / "scratch" / "ref_zh.codes.npy", pipe.generated_codes)
    print("done", sum(len(v) for v in record.values()), "records")


if __name__ == "__main__":
    main()
