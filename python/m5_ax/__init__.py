"""M5-TTS AX650 runtime (numpy + axengine)."""
from .tts import M5AX, read_wav, write_wav

__all__ = ["M5AX", "read_wav", "write_wav"]
