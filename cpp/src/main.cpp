// M5-TTS.AXERA — AX650 C++ runtime: text + reference wav -> 24 kHz speech.
//
// 6 axmodels: codec_encoder, lm_prefill, lm_decode, depth_prefill, depth_step,
//             codec_decoder
// Host side: wav/resample/trim, byte-level BPE (tokenizer.bin), embedding
// lookups, RVQ encode/decode, 15 audio heads + top-k sampling, RoPE/mask/KV.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <sys/stat.h>
#include <unistd.h>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <string>
#include <vector>

#include "audio.hpp"
#include "ax_runner.hpp"
#include "m5_config.h"
#include "tokenizer.hpp"

using m5tts::Tokenizer;
namespace ax = cutetts;

namespace {

struct Args {
    std::string model_dir = "models";
    std::string assets = "assets";
    std::string text = "你好，欢迎使用我们的语音合成模型。";
    std::string prompt_wav = "samples/aslp_default_voice.wav";
    std::string prompt_text = "猪携带该疾病，然后通过蚊子传染给人类。";
    std::string lang = "Chinese";
    std::string output = "out.wav";
    int seed = 1234;
    int max_frames = 400;
    int first_frames = 4;
    int chunk_frames = 8;
    bool stream = false;
    bool greedy = false;
    bool dump_ids = false;
    std::string codes_out;
    std::string ref_out;
    std::string audio_out;
    std::string feat_out;
    std::string decode_codes;
    std::string ref_cache;
    int codebooks = 16;
};

struct Assets {
    ax::Npy text_emb, audio_emb, head0, ahead, proj, proj_audio;
    ax::Npy enc_sem_book, enc_ac_book, enc_sem_w, enc_ac_w;
    ax::Npy dec_sem_book, dec_rest_book, dpsw, dpsb, dprw, dprb, prior, mean_code;
    std::vector<float> bcos, bsin, dcos, dsin, ccos, csin;
};

static bool load_npy(const std::string& dir, const char* name, ax::Npy& out) {
    return ax::read_npy(dir + "/" + name, out);
}

static void rope_tables(const float* inv, float scale, int n, std::vector<float>& cos,
                        std::vector<float>& sin) {
    cos.assign(static_cast<size_t>(n) * 64, 0.f);
    sin.assign(static_cast<size_t>(n) * 64, 0.f);
    for (int p = 0; p < n; ++p) {
        for (int i = 0; i < 32; ++i) {
            const double f = static_cast<double>(p) * inv[i];
            const float c = static_cast<float>(std::cos(f) * scale);
            const float s = static_cast<float>(std::sin(f) * scale);
            cos[static_cast<size_t>(p) * 64 + i] = c;
            cos[static_cast<size_t>(p) * 64 + 32 + i] = c;
            sin[static_cast<size_t>(p) * 64 + i] = s;
            sin[static_cast<size_t>(p) * 64 + 32 + i] = s;
        }
    }
}

static bool load_assets(const std::string& dir, Assets& a) {
    bool ok = load_npy(dir, "text_embeddings.npy", a.text_emb) &&
              load_npy(dir, "audio_embeddings.npy", a.audio_emb) &&
              load_npy(dir, "head0.npy", a.head0) &&
              load_npy(dir, "audio_head.npy", a.ahead) &&
              load_npy(dir, "projection.npy", a.proj) &&
              load_npy(dir, "projected_audio.npy", a.proj_audio) &&
              load_npy(dir, "enc_sem_book.npy", a.enc_sem_book) &&
              load_npy(dir, "enc_ac_book.npy", a.enc_ac_book) &&
              load_npy(dir, "enc_sem_w.npy", a.enc_sem_w) &&
              load_npy(dir, "enc_ac_w.npy", a.enc_ac_w) &&
              load_npy(dir, "dec_sem_book.npy", a.dec_sem_book) &&
              load_npy(dir, "dec_rest_book.npy", a.dec_rest_book) &&
              load_npy(dir, "dec_proj_sem_w.npy", a.dpsw) &&
              load_npy(dir, "dec_proj_sem_b.npy", a.dpsb) &&
              load_npy(dir, "dec_proj_rest_w.npy", a.dprw) &&
              load_npy(dir, "dec_proj_rest_b.npy", a.dprb) &&
              load_npy(dir, "codebook_prior.npy", a.prior) &&
              load_npy(dir, "mean_code.npy", a.mean_code);
    if (!ok) return false;
    // audio_head.npy is [15, 512, 2051] (dim-major); transpose for row-major GEMV
    {
        std::vector<float> t(static_cast<size_t>(15) * 512 * kAudioVocab);
        for (int l = 0; l < 15; ++l)
            for (int d = 0; d < 512; ++d)
                for (int v = 0; v < kAudioVocab; ++v)
                    t[(static_cast<size_t>(l) * kAudioVocab + v) * 512 + d] =
                        a.ahead.data[(static_cast<size_t>(l) * 512 + d) * kAudioVocab + v];
        a.ahead.data.swap(t);
    }
    rope_tables(kBackboneInvFreq, kBackboneRopeScale, kCap + 8, a.bcos, a.bsin);
    rope_tables(kDecoderInvFreq, kDecoderRopeScale, kDepthSlots + 2, a.dcos, a.dsin);
    rope_tables(kCodecInvFreq, 1.f, 2048, a.ccos, a.csin);
    return true;
}

// ---------------------------------------------------------------- host math
static void matvec(const float* W, int rows, int cols, const float* x, float* y) {
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const float* row = W + static_cast<size_t>(r) * cols;
        float acc = 0.f;
        for (int c = 0; c < cols; ++c) acc += row[c] * x[c];
        y[r] = acc;
    }
}

static void add_row(float* dst, const float* src, int n) {
    for (int i = 0; i < n; ++i) dst[i] += src[i];
}

// prompt codes are [16][frames] (codebook-major) -> decoder input is [frames][16]
static void append_frame_major(const std::vector<int>& prompt, int frames, std::vector<int>& out) {
    for (int t = 0; t < frames; ++t)
        for (int cb = 0; cb < 16; ++cb) out.push_back(prompt[static_cast<size_t>(cb) * frames + t]);
}

static int argmin(const float* v, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (v[i] < v[best]) best = i;
    return best;
}

// RVQ encode: feat [512][frames] -> codes [16][frames]
// One GEMM-style pass per codebook (book row read once, residuals stay in L2).
static void rvq_encode(const Assets& a, const float* feat, int frames, int stride, int* codes) {
    const int F = frames;
    std::vector<float> x(static_cast<size_t>(F) * 512);
    for (int t = 0; t < F; ++t)
        for (int c = 0; c < 512; ++c) x[static_cast<size_t>(t) * 512 + c] = feat[static_cast<size_t>(c) * stride + t];
    std::vector<float> sem(static_cast<size_t>(F) * 256), ac(static_cast<size_t>(F) * 256);
    const float* ws = a.enc_sem_w.data.data();
    const float* wa = a.enc_ac_w.data.data();
#pragma omp parallel for schedule(static)
    for (int t = 0; t < F; ++t) {
        for (int r = 0; r < 256; ++r) {
            float acc_s = 0.f, acc_a = 0.f;
            const float* rs = ws + static_cast<size_t>(r) * 512;
            const float* ra = wa + static_cast<size_t>(r) * 512;
            for (int c = 0; c < 512; ++c) {
                const float v = x[static_cast<size_t>(t) * 512 + c];
                acc_s += rs[c] * v;
                acc_a += ra[c] * v;
            }
            sem[static_cast<size_t>(t) * 256 + r] = acc_s;
            ac[static_cast<size_t>(t) * 256 + r] = acc_a;
        }
    }
    std::vector<float> res(static_cast<size_t>(F) * 256), rn(F, 0.f), en(2048, 0.f);
    std::vector<float> dist(static_cast<size_t>(F) * 2048);
    auto nearest = [&](const float* book, int q) {
        const float* src = q == 0 ? sem.data() : ac.data();
        for (int t = 0; t < F; ++t) {
            float n = 0.f;
            for (int c = 0; c < 256; ++c) {
                const float v = q == 0 ? src[static_cast<size_t>(t) * 256 + c]
                                       : (q == 1 ? src[static_cast<size_t>(t) * 256 + c]
                                                 : res[static_cast<size_t>(t) * 256 + c]);
                res[static_cast<size_t>(t) * 256 + c] = v;
                n += v * v;
            }
            rn[t] = n;
        }
#pragma omp parallel for schedule(static)
        for (int k = 0; k < 2048; ++k) {
            const float* e = book + static_cast<size_t>(k) * 256;
            float nn = 0.f;
            for (int c = 0; c < 256; ++c) nn += e[c] * e[c];
            en[k] = nn;
            for (int t = 0; t < F; ++t) {
                const float* r = &res[static_cast<size_t>(t) * 256];
                float dot = 0.f;
                for (int c = 0; c < 256; ++c) dot += r[c] * e[c];
                dist[static_cast<size_t>(t) * 2048 + k] = rn[t] + nn - 2.f * dot;
            }
        }
#pragma omp parallel for schedule(static)
        for (int t = 0; t < F; ++t) {
            const float* d = &dist[static_cast<size_t>(t) * 2048];
            int best = 0;
            for (int k = 1; k < 2048; ++k)
                if (d[k] < d[best]) best = k;
            codes[static_cast<size_t>(q) * F + t] = best;
            const float* e = book + static_cast<size_t>(best) * 256;
            for (int c = 0; c < 256; ++c) res[static_cast<size_t>(t) * 256 + c] -= e[c];
        }
    };
    nearest(a.enc_sem_book.data.data(), 0);
    for (int q = 0; q < 15; ++q)
        nearest(a.enc_ac_book.data.data() + static_cast<size_t>(q) * 2048 * 256, q + 1);
}

// RVQ decode: codes [16][n] -> latent [512][n]
static void rvq_decode(const Assets& a, const int* codes, int n, float* latent, int keep,
                       const float* prior) {
    std::vector<float> sem(256), rest(256);
    for (int t = 0; t < n; ++t) {
        const float* e = a.dec_sem_book.data.data() + static_cast<size_t>(codes[static_cast<size_t>(0) * n + t]) * 256;
        for (int c = 0; c < 256; ++c) sem[c] = e[c];
        std::fill(rest.begin(), rest.end(), 0.f);
        for (int q = 0; q < keep - 1; ++q) {
            const float* er = a.dec_rest_book.data.data() +
                              (static_cast<size_t>(q) * 2048 + static_cast<size_t>(codes[static_cast<size_t>(q + 1) * n + t])) * 256;
            for (int c = 0; c < 256; ++c) rest[c] += er[c];
        }
        for (int o = 0; o < 512; ++o) {
            float v = a.dpsb.data[o] + a.dprb.data[o];
            const float* ws = a.dpsw.data.data() + static_cast<size_t>(o) * 256;
            const float* wr = a.dprw.data.data() + static_cast<size_t>(o) * 256;
            float acc = 0.f;
            for (int c = 0; c < 256; ++c) acc += ws[c] * sem[c] + wr[c] * rest[c];
            latent[static_cast<size_t>(o) * n + t] = v + acc + (prior ? prior[o] : 0.f);
        }
    }
}

// ---------------------------------------------------------------- sampling
struct Rng {
    std::mt19937_64 gen;
    std::uniform_real_distribution<double> uni{1e-12, 1.0};
    explicit Rng(int seed) : gen(static_cast<uint64_t>(seed) * 6364136223846793005ULL + 1442695040888963407ULL) {}
    float exponential() { return static_cast<float>(-std::log(uni(gen))); }
};

static int sample(const float* logits_in, const float* noise, bool allow_eos, bool greedy,
                  std::vector<int>& idx, std::vector<float>& vals) {
    std::vector<float> logits(logits_in, logits_in + kAudioVocab);
    if (!allow_eos) logits[0] = -1e5f;
    // top-k (manual 10-best: ~3x faster than partial_sort for 2051 candidates)
    idx.assign(kTopK, 0);
    float top_val[kTopK];
    for (int i = 0; i < kTopK; ++i) top_val[i] = -1e30f;
    for (int v = 0; v < kAudioVocab; ++v) {
        const float value = logits[v];
        if (value <= top_val[kTopK - 1]) continue;
        int at = kTopK - 1;
        while (at > 0 && top_val[at - 1] < value) { top_val[at] = top_val[at - 1]; idx[at] = idx[at - 1]; --at; }
        top_val[at] = value;
        idx[at] = v;
    }
    vals.resize(kTopK);
    if (greedy) return idx[0];
    float mx = -1e30f;
    for (int i = 0; i < kTopK; ++i) {
        vals[i] = logits[idx[i]] / kTemperature;
        mx = std::max(mx, vals[i]);
    }
    double sum = 0.0;
    for (int i = 0; i < kTopK; ++i) {
        vals[i] = std::exp(vals[i] - mx);
        sum += vals[i];
    }
    int best = 0;
    double best_score = -1.0;
    for (int i = 0; i < kTopK; ++i) {
        const double score = (vals[i] / sum) / noise[idx[i]];
        if (score > best_score) { best_score = score; best = i; }
    }
    return idx[best];
}

// ---------------------------------------------------------- reference cache
// The reference (prompt codes + codec state after the prompt) is independent of
// the target text, so a cache turns TTFA into just prefill + first frames.
struct RefCache {
    std::vector<int> codes;
    std::vector<int> pending;
    std::vector<float> past;
    std::vector<std::vector<float>> states;
    int frames = 0;
    int offset = 0;
};

static uint64_t ref_key(const std::string& path) {
    struct stat st;
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ULL; };
    if (::stat(path.c_str(), &st) == 0) {
        mix(static_cast<uint64_t>(st.st_size));
        mix(static_cast<uint64_t>(st.st_mtime));
    }
    for (unsigned char c : path) mix(c);
    return h;
}

static std::string ref_path(const std::string& dir, const std::string& wav) {
    char name[64];
    std::snprintf(name, sizeof(name), "/ref_%016llx.bin",
                  static_cast<unsigned long long>(ref_key(wav)));
    return dir + name;
}

static void mkdir_p(const std::string& path) {
    std::string acc;
    for (size_t i = 0; i < path.size(); ++i) {
        acc.push_back(path[i]);
        if (path[i] == '/' && acc.size() > 1) ::mkdir(acc.c_str(), 0755);
    }
    if (!path.empty()) ::mkdir(acc.c_str(), 0755);
}

static bool save_ref_cache(const std::string& dir, const std::string& wav,
                           const std::vector<int>& prompt, int frames, int offset,
                           const std::vector<float>& past,
                           const std::vector<std::vector<float>>& states,
                           const std::vector<int>& pending) {
    if (dir.empty()) return false;
    mkdir_p(dir);
    const std::string path = ref_path(dir, wav);
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fwrite("M5REF1\0\0", 1, 8, f);
    const int pending_frames = static_cast<int>(pending.size() / 16);
    const int nstates = static_cast<int>(states.size());
    std::fwrite(&frames, 4, 1, f);
    std::fwrite(&offset, 4, 1, f);
    std::fwrite(&pending_frames, 4, 1, f);
    std::fwrite(&nstates, 4, 1, f);
    std::fwrite(prompt.data(), 4, static_cast<size_t>(frames) * 16, f);
    std::fwrite(pending.data(), 4, pending.size(), f);
    std::fwrite(past.data(), 4, past.size(), f);
    for (const auto& st : states) std::fwrite(st.data(), 4, st.size(), f);
    std::fclose(f);
    fprintf(stderr, "ref cache saved: %s\n", path.c_str());
    return true;
}

static bool load_ref_cache(const std::string& dir, const std::string& wav, RefCache& out) {
    if (dir.empty()) return false;
    FILE* f = std::fopen(ref_path(dir, wav).c_str(), "rb");
    if (!f) return false;
    char magic[8];
    int frames = 0, offset = 0, pending_frames = 0, nstates = 0;
    bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, "M5REF1", 6) == 0 &&
              std::fread(&frames, 4, 1, f) == 1 && std::fread(&offset, 4, 1, f) == 1 &&
              std::fread(&pending_frames, 4, 1, f) == 1 && std::fread(&nstates, 4, 1, f) == 1 &&
              frames > 0 && frames <= 60 && pending_frames >= 0 && pending_frames < kCodecFrames &&
              nstates == kCodecStates;
    if (ok) {
        out.codes.resize(static_cast<size_t>(frames) * 16);
        out.pending.resize(static_cast<size_t>(pending_frames) * 16);
        out.past.resize(static_cast<size_t>(kCodecLayers) * 2 * kCodecKv * (kWindow - 1) * kCodecDim);
        out.states.resize(kCodecStates);
        out.frames = frames;
        out.offset = offset;
        ok = std::fread(out.codes.data(), 4, out.codes.size(), f) == out.codes.size() &&
             std::fread(out.pending.data(), 4, out.pending.size(), f) == out.pending.size() &&
             std::fread(out.past.data(), 4, out.past.size(), f) == out.past.size();
        for (int i = 0; i < kCodecStates && ok; ++i) {
            size_t n = 1;
            for (int d = 0; d < 3; ++d) n *= static_cast<size_t>(kCodecStateShapes[i][d]);
            out.states[i].assign(n, 0.f);
            ok = std::fread(out.states[i].data(), 4, n, f) == n;
        }
    }
    std::fclose(f);
    return ok;
}

// ---------------------------------------------------------------- pipeline
struct Pipeline {
    Args args;
    Assets a;
    Tokenizer tok;
    ax::AxModel prefill_model, decode_model, depth_pre_model, depth_step_model,
        enc_model, dec_model;
    std::vector<float> kv;        // [8][2][2][512][64]
    std::vector<float> depth_kv;  // [5][2][2][16][64]
    std::vector<float> codec_past;
    std::vector<std::vector<float>> codec_states;
    std::vector<float> codec_pending;  // frames x 16
    int codec_offset = 0;

    bool init() {
        if (!load_assets(args.assets, a)) return false;
        if (!tok.load(args.assets + "/tokenizer.bin")) return false;
        const std::string m = args.model_dir + "/";
        if (!prefill_model.load(m + "lm_prefill.axmodel")) return false;
        if (!decode_model.load(m + "lm_decode.axmodel")) return false;
        if (!depth_pre_model.load(m + "depth_prefill.axmodel")) return false;
        if (!depth_step_model.load(m + "depth_step.axmodel")) return false;
        if (!enc_model.load(m + "codec_encoder.axmodel")) return false;
        if (!dec_model.load(m + "codec_decoder.axmodel")) return false;
        kv.assign(static_cast<size_t>(kBackboneLayers) * 2 * kBackboneKv * kCap * 64, 0.f);
        depth_kv.assign(static_cast<size_t>(kDecoderLayers) * 2 * kDecoderKv * kDepthSlots * 64, 0.f);
        codec_past.assign(static_cast<size_t>(kCodecLayers) * 2 * kCodecKv * (kWindow - 1) * kCodecDim, 0.f);
        prior_keep.assign(512, 0.f);
        codec_states.resize(kCodecStates);
        for (int i = 0; i < kCodecStates; ++i) {
            size_t n = 1;
            for (int d = 0; d < 3; ++d) n *= static_cast<size_t>(kCodecStateShapes[i][d]);
            codec_states[i].assign(n, 0.f);
        }
        return true;
    }

    // ---------------------------------------------------------- reference
    bool encode_reference(const std::string& wav_path, int* codes, int& frames) {
        std::vector<float> audio;
        int sr = 0;
        if (!ax::read_wav(wav_path, audio, sr)) return false;
        audio = ax::resample_to(audio, sr, kSampleRate);
        float peak = 1e-8f;
        for (float v : audio) peak = std::max(peak, std::fabs(v));
        for (float& v : audio) v /= peak;
        // remove long silences (same rule as the Python runtime)
        const int n = static_cast<int>(audio.size());
        const float threshold = (n > 10 * kSampleRate) ? 0.02f : 0.015f;
        std::vector<int> speech;
        for (int i = 0; i < n; ++i)
            if (std::fabs(audio[i]) > threshold) speech.push_back(i);
        if (!speech.empty()) {
            const int gap = static_cast<int>(0.15 * kSampleRate);
            std::vector<std::pair<int, int>> segs;
            int start = speech[0], prev = speech[0];
            for (size_t i = 1; i < speech.size(); ++i) {
                if (speech[i] - prev > gap) {
                    segs.emplace_back(start, prev + 1);
                    start = speech[i];
                }
                prev = speech[i];
            }
            const int tail = n - speech.back() - 1;
            segs.emplace_back(start, (tail >= gap) ? speech.back() + 1 : n);
            std::vector<float> merged;
            for (auto& s : segs) {
                const int lo = std::max(0, s.first - 1200);
                const int hi = std::min(n, s.second + 1200);
                merged.insert(merged.end(), audio.begin() + lo, audio.begin() + hi);
            }
            audio.swap(merged);
        }
        std::vector<float> wave(kCodecSamples, 0.f);
        const int use = std::min<int>(static_cast<int>(audio.size()), kCodecSamples);
        std::memcpy(wave.data(), audio.data(), sizeof(float) * use);
        if (!args.audio_out.empty()) {
            FILE* af = std::fopen(args.audio_out.c_str(), "wb");
            if (af) { std::fwrite(wave.data(), sizeof(float), wave.size(), af); std::fclose(af); }
        }
        enc_model.set("audio", wave.data(), wave.size());
        if (!enc_model.run()) return false;
        std::vector<float> feat(static_cast<size_t>(512) * 60);
        if (!enc_model.get("codes", feat.data(), feat.size())) return false;
        if (!args.feat_out.empty()) {
            FILE* ff = std::fopen(args.feat_out.c_str(), "wb");
            if (ff) { std::fwrite(feat.data(), sizeof(float), feat.size(), ff); std::fclose(ff); }
        }
        frames = std::max(1, static_cast<int>(std::ceil(static_cast<double>(use) / kHop)));
        rvq_encode(a, feat.data(), frames, kCodecSamples / kHop, codes);
        return true;
    }

    // -------------------------------------------------------- codec stream
    int codebooks = 16;
    std::vector<float> prior_keep;

    void decode_block(const int* codes, int n, bool final_block, std::vector<float>& pcm) {
        std::vector<float> latent(static_cast<size_t>(512) * kCodecFrames, 0.f);
        std::vector<int> icodes(static_cast<size_t>(n) * 16);
        for (int t = 0; t < n; ++t)
            for (int cb = 0; cb < 16; ++cb) icodes[static_cast<size_t>(cb) * n + t] = codes[t * 16 + cb];
        std::vector<float> full(static_cast<size_t>(512) * n);
        rvq_decode(a, icodes.data(), n, full.data(), codebooks, prior_keep.data());
        for (int c = 0; c < 512; ++c)
            for (int t = 0; t < n; ++t) latent[static_cast<size_t>(c) * kCodecFrames + t] = full[static_cast<size_t>(c) * n + t];
        std::vector<float> mask(static_cast<size_t>(kCodecFrames) * (kWindow - 1 + kCodecFrames));
        for (int t = 0; t < kCodecFrames; ++t) {
            const int pos = codec_offset + t;
            for (int j = 0; j < kWindow - 1 + kCodecFrames; ++j) {
                const int key_abs = j + codec_offset - (kWindow - 1);
                const bool ok = key_abs <= pos && key_abs >= 0 && key_abs >= pos - (kWindow - 1);
                mask[static_cast<size_t>(t) * (kWindow - 1 + kCodecFrames) + j] = ok ? 0.f : -30.f;
            }
        }
        std::vector<float> cos(static_cast<size_t>(kCodecFrames) * 64), sin(static_cast<size_t>(kCodecFrames) * 64);
        for (int t = 0; t < kCodecFrames; ++t) {
            std::memcpy(&cos[static_cast<size_t>(t) * 64], &a.ccos[static_cast<size_t>(codec_offset + t) * 64], 64 * sizeof(float));
            std::memcpy(&sin[static_cast<size_t>(t) * 64], &a.csin[static_cast<size_t>(codec_offset + t) * 64], 64 * sizeof(float));
        }
        dec_model.set("latent", latent);
        dec_model.set("past", codec_past);
        dec_model.set("cos", cos);
        dec_model.set("sin", sin);
        dec_model.set("mask", mask);
        for (int i = 0; i < kCodecStates; ++i) dec_model.set("conv_" + std::to_string(i), codec_states[i]);
        if (!dec_model.run()) return;
        dec_model.get("past_out", codec_past.data(), codec_past.size());
        for (int i = 0; i < kCodecStates; ++i) dec_model.get("conv_out_" + std::to_string(i), codec_states[i].data(), codec_states[i].size());
        std::vector<float> audio(static_cast<size_t>(kCodecFrames) * kHop);
        dec_model.get("audio", audio.data(), audio.size());
        pcm.assign(audio.begin(), audio.begin() + static_cast<size_t>(n) * kHop);
        codec_offset += n;
    }

    void push_codes(const std::vector<int>& frames16, bool final_block, std::vector<float>& pcm) {
        codec_pending.insert(codec_pending.end(), frames16.begin(), frames16.end());
        pcm.clear();
        const int pending_frames = static_cast<int>(codec_pending.size() / 16);
        int offset = 0;
        while (pending_frames - offset >= kCodecFrames || (final_block && pending_frames - offset > 0)) {
            const int n = std::min(kCodecFrames, pending_frames - offset);
            std::vector<int> block_int(kCodecFrames * 16, 0);
            for (int i = 0; i < n * 16; ++i) block_int[i] = static_cast<int>(codec_pending[static_cast<size_t>(offset) * 16 + i]);
            std::vector<float> out;
            decode_block(block_int.data(), n, final_block, out);
            pcm.insert(pcm.end(), out.begin(), out.end());
            offset += n;
        }
        codec_pending.erase(codec_pending.begin(), codec_pending.begin() + static_cast<long>(offset) * 16);
    }

    // ------------------------------------------------------------ generate
    bool run() {
        const auto t_start = std::chrono::steady_clock::now();
        codebooks = std::max(1, std::min(16, args.codebooks));
        std::fill(prior_keep.begin(), prior_keep.end(), 0.f);
        for (int q = codebooks; q < 16; ++q)
            for (int c = 0; c < 512; ++c) prior_keep[c] += a.prior.data[static_cast<size_t>(q) * 512 + c];
        double codec_enc_s = 0.0, prefill_s = 0.0, ar_s = 0.0, codec_s = 0.0;
        auto now = [](std::chrono::steady_clock::time_point t) {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
        };
        std::vector<int> prompt(static_cast<size_t>(16) * 60);
        int prompt_frames = 0;
        RefCache refcache;
        const std::string cache_dir = args.ref_cache.empty() ? std::string()
            : args.ref_cache + "/cb" + std::to_string(codebooks);
        const bool cached = load_ref_cache(cache_dir, args.prompt_wav, refcache);
        auto t0 = std::chrono::steady_clock::now();
        if (cached) {
            prompt_frames = refcache.frames;
            prompt.assign(refcache.codes.begin(), refcache.codes.end());
            codec_past = refcache.past;
            codec_states = refcache.states;
            codec_offset = refcache.offset;
            codec_pending.assign(refcache.pending.size(), 0.f);
            for (size_t i = 0; i < refcache.pending.size(); ++i)
                codec_pending[i] = static_cast<float>(refcache.pending[i]);
        } else if (!encode_reference(args.prompt_wav, prompt.data(), prompt_frames)) {
            return false;
        }
        codec_enc_s = now(t0);
        fprintf(stderr, "reference: %d frames (%.2fs audio)\n", prompt_frames, prompt_frames * kHop / 24000.0);

        if (!args.ref_out.empty()) {
            FILE* rf = std::fopen(args.ref_out.c_str(), "wb");
            if (rf) {
                for (int t = 0; t < prompt_frames; ++t)
                    for (int c = 0; c < 16; ++c)
                        std::fprintf(rf, "%d%s", prompt[static_cast<size_t>(c) * prompt_frames + t], c == 15 ? "\n" : ",");
                std::fclose(rf);
            }
        }
        if (!args.decode_codes.empty()) {   // decode-only mode (A/B against Python)
            std::vector<int> extra;
            FILE* cf = std::fopen(args.decode_codes.c_str(), "rb");
            if (!cf) return false;
            char line[512];
            while (std::fgets(line, sizeof(line), cf)) {
                int values[16];
                if (std::sscanf(line, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
                                &values[0], &values[1], &values[2], &values[3], &values[4],
                                &values[5], &values[6], &values[7], &values[8], &values[9],
                                &values[10], &values[11], &values[12], &values[13],
                                &values[14], &values[15]) == 16) {
                    for (int v : values) extra.push_back(v);
                }
            }
            std::fclose(cf);
            std::vector<int> all;
            append_frame_major(prompt, prompt_frames, all);
            all.insert(all.end(), extra.begin(), extra.end());
            std::vector<float> pcm;
            push_codes(all, true, pcm);
            const size_t drop = static_cast<size_t>(prompt_frames) * kHop;
            std::vector<float> wav_out(pcm.begin() + static_cast<long>(std::min(drop, pcm.size())), pcm.end());
            if (!ax::write_wav(args.output, wav_out, kSampleRate)) return false;
            fprintf(stderr, "decoded %zu codes -> %.2fs -> %s\n", extra.size() / 16,
                    wav_out.size() / static_cast<double>(kSampleRate), args.output.c_str());
            return true;
        }
        const std::string sentence =
            "<|caption_begin|><|caption_end|><|accent_begin|><|accent_end|>"
            "<|speaker_start|>None<|speaker_end|><|lang_begin|>" + args.lang +
            "<|lang_end|>" + args.prompt_text + args.text;
        std::vector<int> ids = tok.encode(sentence);
        const int real = static_cast<int>(ids.size()) + prompt_frames;
        if (real >= kSeq) { fprintf(stderr, "prompt too long: %d\n", real); return false; }

        std::vector<float> hidden(static_cast<size_t>(kSeq) * kHidden, 0.f);
        for (size_t i = 0; i < ids.size(); ++i)
            std::memcpy(&hidden[i * kHidden], &a.text_emb.data[static_cast<size_t>(ids[i]) * kHidden],
                        kHidden * sizeof(float));
        for (int t = 0; t < prompt_frames; ++t) {
            float* dst = &hidden[static_cast<size_t>(ids.size() + t) * kHidden];
            for (int cb = 0; cb < 16; ++cb) {
                const int code = prompt[static_cast<size_t>(cb) * prompt_frames + t] + 1 + cb * kAudioVocab;
                add_row(dst, &a.audio_emb.data[static_cast<size_t>(code) * kHidden], kHidden);
            }
        }
        for (int i = real; i < kSeq; ++i)
            std::memcpy(&hidden[static_cast<size_t>(i) * kHidden], &hidden[static_cast<size_t>(real - 1) * kHidden],
                        kHidden * sizeof(float));

        t0 = std::chrono::steady_clock::now();
        prefill_model.set("hidden", hidden);
        prefill_model.set("cos", a.bcos.data(), static_cast<size_t>(kSeq) * 64);
        prefill_model.set("sin", a.bsin.data(), static_cast<size_t>(kSeq) * 64);
        if (!prefill_model.run()) return false;
        std::vector<float> hout(static_cast<size_t>(kSeq) * kHidden);
        prefill_model.get("hidden_out", hout.data(), hout.size());
        std::vector<float> pkv(static_cast<size_t>(kBackboneLayers) * 2 * kBackboneKv * kSeq * 64);
        prefill_model.get("kv", pkv.data(), pkv.size());
        for (int l = 0; l < kBackboneLayers; ++l)
            for (int kvd = 0; kvd < 2; ++kvd)
                for (int h = 0; h < kBackboneKv; ++h)
                    std::memcpy(&kv[((static_cast<size_t>(l) * 2 + kvd) * kBackboneKv + h) * kCap * 64],
                                &pkv[((static_cast<size_t>(l) * 2 + kvd) * kBackboneKv + h) * kSeq * 64],
                                static_cast<size_t>(real) * 64 * sizeof(float));
        prefill_s = now(t0);
        std::vector<float> last(&hout[static_cast<size_t>(real - 1) * kHidden],
                                &hout[static_cast<size_t>(real) * kHidden]);
        int pos = real;

        double ttft = -1.0, ttfa = -1.0;
        std::vector<float> wav;
        ax::WavWriter writer;
        const size_t prompt_samples = static_cast<size_t>(prompt_frames) * kHop;
        // when the cache holds the prompt state, only its leftover pending frames remain
        size_t skip = cached ? (codec_pending.size() / 16) * kHop : prompt_samples;
        if (args.stream && !writer.open(args.output, kSampleRate)) return false;
        auto consume = [&](std::vector<float>& pcm) {
            if (skip && !pcm.empty()) {
                const size_t k = std::min(skip, pcm.size());
                pcm.erase(pcm.begin(), pcm.begin() + static_cast<long>(k));
                skip -= k;
            }
            if (args.stream) {
                if (!pcm.empty()) {
                    writer.write(pcm.data(), pcm.size());
                    if (ttfa < 0) ttfa = now(t_start);
                }
            } else {
                wav.insert(wav.end(), pcm.begin(), pcm.end());
            }
        };
        if (!cached) {
            std::vector<int> pc;
            append_frame_major(prompt, prompt_frames, pc);
            std::vector<float> pcm;
            auto tc = std::chrono::steady_clock::now();
            push_codes(pc, false, pcm);
            codec_s += now(tc);
            consume(pcm);
            if (!cache_dir.empty()) {
                std::vector<int> pending;
                for (float v : codec_pending) pending.push_back(static_cast<int>(v));
                save_ref_cache(cache_dir, args.prompt_wav, prompt, prompt_frames,
                               codec_offset, codec_past, codec_states, pending);
            }
        }

        Rng rng(args.seed);
        std::vector<float> noise(static_cast<size_t>(16) * kAudioVocab);
        std::vector<int> gen;  // frames*16, codes minus one
        std::vector<float> mask(static_cast<size_t>(1) * (kCap + 1));
        std::vector<float> dlocal(static_cast<size_t>(2) * kHidden);
        std::vector<float> depth_mask(kDepthSlots + 1);
        std::vector<int> idx;
        std::vector<float> vals;
        auto t_ar = std::chrono::steady_clock::now();
        int frames = 0;
        bool eos = false;
        const int frame_limit = std::min(args.max_frames, kCap - pos);
        int queued = 0, threshold = args.first_frames + ((kCodecFrames - prompt_frames % kCodecFrames) % kCodecFrames);
        bool first_audio = true;

        for (int frame = 0; frame < frame_limit; ++frame) {
            if (!args.greedy)
                for (int i = 0; i < 16; ++i)
                    for (int j = 0; j < kAudioVocab; ++j) noise[static_cast<size_t>(i) * kAudioVocab + j] = rng.exponential();
            std::vector<float> logits(kAudioVocab);
            matvec(a.head0.data.data(), kAudioVocab, kHidden, last.data(), logits.data());
            const int c0 = sample(logits.data(), &noise[0], frame > kMinFrames, args.greedy, idx, vals);
            std::vector<int> codes(16, 0);
            codes[0] = c0;
            matvec(a.proj.data.data(), kHidden, kHidden, last.data(), &dlocal[0]);
            matvec(a.proj.data.data(), kHidden, kHidden, &a.audio_emb.data[static_cast<size_t>(c0) * kHidden], &dlocal[kHidden]);
            depth_pre_model.set("local", dlocal);
            if (!depth_pre_model.run()) return false;
            std::vector<float> dh(kHidden);
            depth_pre_model.get("hidden_out", dh.data(), dh.size());
            std::vector<float> dkv(static_cast<size_t>(kDecoderLayers) * 2 * kDecoderKv * 2 * 64);
            depth_pre_model.get("kv", dkv.data(), dkv.size());
            for (int l = 0; l < kDecoderLayers; ++l)
                for (int kvd = 0; kvd < 2; ++kvd)
                    for (int h = 0; h < kDecoderKv; ++h)
                        std::memcpy(&depth_kv[((static_cast<size_t>(l) * 2 + kvd) * kDecoderKv + h) * kDepthSlots * 64],
                                    &dkv[((static_cast<size_t>(l) * 2 + kvd) * kDecoderKv + h) * 2 * 64],
                                    2 * 64 * sizeof(float));
            std::vector<float> h = dh;
            for (int level = 0; level < codebooks - 1; ++level) {
                matvec(a.ahead.data.data() + static_cast<size_t>(level) * kAudioVocab * kHidden,
                       kAudioVocab, kHidden, h.data(), logits.data());
                const int ci = sample(logits.data(), &noise[static_cast<size_t>(level + 1) * kAudioVocab], true,
                                      args.greedy, idx, vals);
                codes[level + 1] = ci;
                if (level == codebooks - 2) break;
                const int token_pos = level + 2;
                std::memcpy(&dlocal[0], &a.proj_audio.data[static_cast<size_t>(ci + (level + 1) * kAudioVocab) * kHidden],
                            kHidden * sizeof(float));
                std::fill(depth_mask.begin(), depth_mask.end(), -30.f);
                for (int i = 0; i < token_pos; ++i) depth_mask[i] = 0.f;
                depth_mask[kDepthSlots] = 0.f;
                std::vector<float> dcos(64), dsin(64);
                std::memcpy(dcos.data(), &a.dcos[static_cast<size_t>(token_pos) * 64], 64 * sizeof(float));
                std::memcpy(dsin.data(), &a.dsin[static_cast<size_t>(token_pos) * 64], 64 * sizeof(float));
                depth_step_model.set("local", &dlocal[0], kHidden);
                depth_step_model.set("cos", dcos);
                depth_step_model.set("sin", dsin);
                depth_step_model.set("mask", depth_mask);
                depth_step_model.set("past", depth_kv);
                if (!depth_step_model.run()) return false;
                depth_step_model.get("hidden_out", h.data(), h.size());
                std::vector<float> nkv(static_cast<size_t>(kDecoderLayers) * 2 * kDecoderKv * 64);
                depth_step_model.get("kv", nkv.data(), nkv.size());
                for (int l = 0; l < kDecoderLayers; ++l)
                    for (int kvd = 0; kvd < 2; ++kvd)
                        for (int hh = 0; hh < kDecoderKv; ++hh)
                            std::memcpy(&depth_kv[(((static_cast<size_t>(l) * 2 + kvd) * kDecoderKv + hh) * kDepthSlots + token_pos) * 64],
                                        &nkv[((static_cast<size_t>(l) * 2 + kvd) * kDecoderKv + hh) * 64],
                                        64 * sizeof(float));
            }
            if (ttft < 0) ttft = now(t_start);
            bool all_zero = true;
            for (int c : codes) all_zero = all_zero && c == 0;
            if (all_zero) { eos = true; break; }
            for (int cb = 0; cb < 16; ++cb) gen.push_back(std::max(0, std::min(kAudioVocab - 2, codes[cb] - 1)));
            ++frames;
            std::vector<float> next(kHidden, 0.f);
            for (int cb = 0; cb < codebooks; ++cb) {
                const int code = codes[cb] + cb * kAudioVocab;
                add_row(next.data(), &a.audio_emb.data[static_cast<size_t>(code) * kHidden], kHidden);
            }
            for (int c = 0; c < kHidden; ++c) next[c] += prior_keep[c];
            std::fill(mask.begin(), mask.end(), -30.f);
            for (int i = 0; i < pos; ++i) mask[i] = 0.f;
            mask[kCap] = 0.f;
            std::vector<float> bcos(64), bsin(64);
            std::memcpy(bcos.data(), &a.bcos[static_cast<size_t>(pos) * 64], 64 * sizeof(float));
            std::memcpy(bsin.data(), &a.bsin[static_cast<size_t>(pos) * 64], 64 * sizeof(float));
            decode_model.set("hidden", next);
            decode_model.set("cos", bcos);
            decode_model.set("sin", bsin);
            decode_model.set("mask", mask);
            decode_model.set("past", kv);
            if (!decode_model.run()) return false;
            decode_model.get("hidden_out", last.data(), last.size());
            std::vector<float> nkv(static_cast<size_t>(kBackboneLayers) * 2 * kBackboneKv * 64);
            decode_model.get("kv", nkv.data(), nkv.size());
            for (int l = 0; l < kBackboneLayers; ++l)
                for (int kvd = 0; kvd < 2; ++kvd)
                    for (int hh = 0; hh < kBackboneKv; ++hh)
                        std::memcpy(&kv[(((static_cast<size_t>(l) * 2 + kvd) * kBackboneKv + hh) * kCap + pos) * 64],
                                    &nkv[((static_cast<size_t>(l) * 2 + kvd) * kBackboneKv + hh) * 64],
                                    64 * sizeof(float));
            ++pos;
            if (args.stream && frames - queued >= threshold) {
                std::vector<int> block(gen.begin() + static_cast<long>(queued) * 16, gen.end());
                std::vector<float> pcm;
                auto tc = std::chrono::steady_clock::now();
                push_codes(block, false, pcm);
                codec_s += now(tc);
                queued = frames;
                threshold = args.chunk_frames;
                consume(pcm);
            }
        }
        ar_s = now(t_ar);
        auto t_codec = std::chrono::steady_clock::now();
        if (frames > queued) {
            std::vector<int> block(gen.begin() + static_cast<long>(queued) * 16, gen.end());
            std::vector<float> pcm;
            push_codes(block, false, pcm);
            consume(pcm);
        }
        {
            std::vector<float> pcm;
            push_codes({}, true, pcm);
            consume(pcm);
        }
        codec_s += now(t_codec);
        if (args.stream) {
            writer.close();
        } else if (!ax::write_wav(args.output, wav, kSampleRate)) {
            return false;
        }
        if (!args.codes_out.empty()) {
            FILE* cf = std::fopen(args.codes_out.c_str(), "wb");
            if (cf) {
                for (int f = 0; f < frames; ++f)
                    for (int c = 0; c < 16; ++c) std::fprintf(cf, "%d%s", gen[static_cast<size_t>(f) * 16 + c], c == 15 ? "\n" : ",");
                std::fclose(cf);
            }
        }
        const double total = now(t_start);
        const double seconds = static_cast<double>(frames) * kHop / kSampleRate;
        fprintf(stderr,
                "frames %d eos %d audio %.2fs total %.2fs RTF %.2f | enc %.3fs prefill %.3fs ar %.3fs codec %.3fs | ttft %.2fs ttfa %.2fs\n",
                frames, eos ? 1 : 0, seconds, total, total / std::max(seconds, 1e-6), codec_enc_s,
                prefill_s, ar_s, codec_s, ttft, ttfa);
        fprintf(stderr, "saved %s\n", args.output.c_str());
        return true;
    }
};

}  // namespace

int main(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", name); std::exit(2); }
            return std::string(argv[++i]);
        };
        if (a == "--model-dir") args.model_dir = next("--model-dir");
        else if (a == "--assets") args.assets = next("--assets");
        else if (a == "--text") args.text = next("--text");
        else if (a == "--prompt-wav") args.prompt_wav = next("--prompt-wav");
        else if (a == "--prompt-text") args.prompt_text = next("--prompt-text");
        else if (a == "--lang") args.lang = next("--lang");
        else if (a == "--output") args.output = next("--output");
        else if (a == "--seed") args.seed = std::stoi(next("--seed"));
        else if (a == "--max-frames") args.max_frames = std::stoi(next("--max-frames"));
        else if (a == "--first-frames") args.first_frames = std::stoi(next("--first-frames"));
        else if (a == "--chunk-frames") args.chunk_frames = std::stoi(next("--chunk-frames"));
        else if (a == "--codes-out") args.codes_out = next("--codes-out");
        else if (a == "--ref-out") args.ref_out = next("--ref-out");
        else if (a == "--audio-out") args.audio_out = next("--audio-out");
        else if (a == "--feat-out") args.feat_out = next("--feat-out");
        else if (a == "--decode-codes") args.decode_codes = next("--decode-codes");
        else if (a == "--ref-cache") args.ref_cache = next("--ref-cache");
        else if (a == "--codebooks") args.codebooks = std::stoi(next("--codebooks"));
        else if (a == "--stream") args.stream = true;
        else if (a == "--greedy") args.greedy = true;
        else if (a == "--dump-ids") { args.dump_ids = true; args.output.clear(); }
        else if (a == "-h" || a == "--help") {
            std::printf("usage: %s [--model-dir models] [--assets assets] --text ... --prompt-wav ... "
                        "--prompt-text ... [--lang Chinese] [--seed 1234] [--stream] [--greedy] [--output out.wav]\n",
                        argv[0]);
            return 0;
        } else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (args.dump_ids) {
        Tokenizer tok;
        if (!tok.load(args.assets + "/tokenizer.bin")) return 1;
        const std::string sentence =
            "<|caption_begin|><|caption_end|><|accent_begin|><|accent_end|>"
            "<|speaker_start|>None<|speaker_end|><|lang_begin|>" + args.lang +
            "<|lang_end|>" + args.prompt_text + args.text;
        auto ids = tok.encode(sentence);
        for (size_t i = 0; i < ids.size(); ++i) printf("%d%s", ids[i], i + 1 == ids.size() ? "\n" : ",");
        return 0;
    }
    Pipeline pipe;
    pipe.args = args;
    if (!pipe.init()) return 1;
    return pipe.run() ? 0 : 1;
}
