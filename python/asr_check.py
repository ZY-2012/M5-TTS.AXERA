#!/usr/bin/env python3
"""Transcribe wav files with the board's SenseVoice axmodel (24 kHz input ok)."""
import sys
import numpy as np
import soundfile as sf

APP = "/root/huan/"
app_dir = "/root/huyuan/183_workspace/Speech_Translation/app"
sys.path.insert(0, app_dir)
from asr_sensevoice import SenseVoiceASR


def main():
    asr = SenseVoiceASR({"asr": {"offline_model_dir":
                                 "/root/huyuan/183_workspace/Speech_Translation/models/asr_offline",
                                 "offline_language": "auto", "withitn": True}})
    asr.load()
    for path in sys.argv[1:]:
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim == 2:
            wav = wav.mean(1)
        if sr != 16000:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, 16000)
            wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)
        print(path, "->", asr.transcribe(wav), flush=True)


if __name__ == "__main__":
    main()
