#!/bin/bash
# M5-TTS.AXERA one-click demo (AX650).
#   bash run_ax650.sh "text" [reference.wav] [prompt-text] [output.wav] [--stream]
#                     [--python] [--seed 1234] [--max-frames N]
set -e
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/soc/lib:${LD_LIBRARY_PATH:-}
export PYTHONDONTWRITEBYTECODE=1

TEXT="${1:-你好，欢迎使用我们的语音合成模型。}"
REF="${2:-samples/aslp_default_voice.wav}"
PROMPT="${3:-猪携带该疾病，然后通过蚊子传染给人类。}"
OUT="${4:-out.wav}"
if [ $# -ge 4 ]; then shift 4; elif [ $# -ge 3 ]; then shift 3; elif [ $# -ge 2 ]; then shift 2; elif [ $# -ge 1 ]; then shift 1; fi

MODE="cpp"
EXTRA=()
CACHE="cache/ref"
for a in "$@"; do
  case "$a" in
    --python) MODE="python" ;;
    --cpp) MODE="cpp" ;;
    --no-cache) CACHE="" ;;
    *) EXTRA+=("$a") ;;
  esac
done
if [ ! -x ./bin/m5tts_ax650 ]; then MODE="python"; fi

if [ "$MODE" = "cpp" ]; then
  exec ./bin/m5tts_ax650 --model-dir models --assets models/assets --ref-cache "$CACHE" \
      --text "$TEXT" --prompt-wav "$REF" --prompt-text "$PROMPT" --output "$OUT" "${EXTRA[@]}"
else
  exec python3 python/demo.py --model-dir models --assets models/assets \
      --text "$TEXT" --prompt-wav "$REF" --prompt-text "$PROMPT" --output "$OUT" "${EXTRA[@]}"
fi
