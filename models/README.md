# Models

The 6 axmodels and the host assets are published on HuggingFace:

    https://huggingface.co/HY-2012/M5-TTS.AXERA

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
| `codec_decoder.axmodel` | 116 MB | 8 latent frames -> 15360 samples |
| `assets/` | 530 MB | embeddings, heads, RVQ tables, tokenizer |
