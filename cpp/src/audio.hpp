#pragma once
// WAV I/O, polyphase resampling, float32/float16 .npy reader, radix-2 FFT and the
// exact log-mel front end used by the speaker encoder.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace cutetts {

// ------------------------------------------------------------------ WAV I/O
static inline bool read_wav(const std::string& path, std::vector<float>& out, int& sr) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open wav %s\n", path.c_str());
        return false;
    }
    char riff[4], wave[4];
    uint32_t size = 0;
    if (std::fread(riff, 1, 4, f) != 4 || std::fread(&size, 4, 1, f) != 1 ||
        std::fread(wave, 1, 4, f) != 4 || std::memcmp(riff, "RIFF", 4) || std::memcmp(wave, "WAVE", 4)) {
        std::fprintf(stderr, "not a RIFF/WAVE file: %s\n", path.c_str());
        std::fclose(f);
        return false;
    }
    uint16_t fmt = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    std::vector<uint8_t> data;
    while (true) {
        char id[4];
        uint32_t sz = 0;
        if (std::fread(id, 1, 4, f) != 4 || std::fread(&sz, 4, 1, f) != 1) break;
        if (!std::memcmp(id, "fmt ", 4)) {
            std::vector<uint8_t> buf(sz);
            if (std::fread(buf.data(), 1, sz, f) != sz) break;
            std::memcpy(&fmt, buf.data(), 2);
            std::memcpy(&channels, buf.data() + 2, 2);
            std::memcpy(&rate, buf.data() + 4, 4);
            std::memcpy(&bits, buf.data() + 14, 2);
        } else if (!std::memcmp(id, "data", 4)) {
            data.resize(sz);
            if (std::fread(data.data(), 1, sz, f) != sz) break;
        } else {
            fseek(f, sz, SEEK_CUR);
        }
    }
    std::fclose(f);
    sr = static_cast<int>(rate);
    if (channels == 0) channels = 1;
    size_t frames = 0;
    std::vector<float> mono;
    if (bits == 16) {
        frames = data.size() / 2 / channels;
        mono.resize(frames);
        const int16_t* p = reinterpret_cast<const int16_t*>(data.data());
        for (size_t i = 0; i < frames; ++i) {
            float acc = 0.f;
            for (int c = 0; c < channels; ++c) acc += p[i * channels + c];
            mono[i] = acc / channels / 32768.f;
        }
    } else if (bits == 32) {
        frames = data.size() / 4 / channels;
        mono.resize(frames);
        const int32_t* p = reinterpret_cast<const int32_t*>(data.data());
        for (size_t i = 0; i < frames; ++i) {
            float acc = 0.f;
            for (int c = 0; c < channels; ++c) acc += p[i * channels + c];
            mono[i] = acc / channels / 2147483648.f;
        }
    } else {
        std::fprintf(stderr, "unsupported wav bit depth %u\n", bits);
        return false;
    }
    out.swap(mono);
    return true;
}

static inline bool write_wav(const std::string& path, const std::vector<float>& x, int sr) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t n = static_cast<uint32_t>(x.size());
    const uint32_t data_bytes = n * 2;
    uint32_t riff = 36 + data_bytes, fmt_size = 16;
    uint16_t fmt = 1, ch = 1, align = 2, bits = 16;
    uint32_t rate = static_cast<uint32_t>(sr), byte_rate = rate * 2;
    std::fwrite("RIFF", 1, 4, f); std::fwrite(&riff, 4, 1, f); std::fwrite("WAVE", 1, 4, f);
    std::fwrite("fmt ", 1, 4, f); std::fwrite(&fmt_size, 4, 1, f);
    std::fwrite(&fmt, 2, 1, f); std::fwrite(&ch, 2, 1, f); std::fwrite(&rate, 4, 1, f);
    std::fwrite(&byte_rate, 4, 1, f); std::fwrite(&align, 2, 1, f); std::fwrite(&bits, 2, 1, f);
    std::fwrite("data", 1, 4, f); std::fwrite(&data_bytes, 4, 1, f);
    std::vector<int16_t> pcm(n);
    for (uint32_t i = 0; i < n; ++i) {
        float v = std::max(-1.f, std::min(1.f, x[i]));
        pcm[i] = static_cast<int16_t>(std::lround(v * 32767.f));
    }
    std::fwrite(pcm.data(), 2, n, f);
    std::fclose(f);
    return true;
}

// ------------------------------------------------------- streaming WAV writer
class WavWriter {
public:
    bool open(const std::string& path, int sr) {
        f_ = std::fopen(path.c_str(), "wb");
        if (!f_) return false;
        sr_ = sr;
        uint8_t zero[44] = {0};
        std::fwrite(zero, 1, 44, f_);
        frames_ = 0;
        return true;
    }
    bool write(const float* x, size_t n) {
        if (!f_ || n == 0) return true;
        std::vector<int16_t> pcm(n);
        for (size_t i = 0; i < n; ++i) {
            float v = std::max(-1.f, std::min(1.f, x[i]));
            pcm[i] = static_cast<int16_t>(std::lround(v * 32767.f));
        }
        std::fwrite(pcm.data(), 2, n, f_);
        frames_ += n;
        return true;
    }
    bool close() {
        if (!f_) return false;
        const uint32_t data_bytes = static_cast<uint32_t>(frames_ * 2);
        const uint32_t riff = 36 + data_bytes;
        const uint32_t rate = static_cast<uint32_t>(sr_);
        const uint16_t fmt = 1, ch = 1, align = 2, bits = 16;
        const uint32_t byte_rate = rate * 2;
        std::fseek(f_, 0, SEEK_SET);
        std::fwrite("RIFF", 1, 4, f_);
        std::fwrite(&riff, 4, 1, f_);
        std::fwrite("WAVEfmt ", 1, 8, f_);
        uint32_t fmt_size = 16;
        std::fwrite(&fmt_size, 4, 1, f_);
        std::fwrite(&fmt, 2, 1, f_);
        std::fwrite(&ch, 2, 1, f_);
        std::fwrite(&rate, 4, 1, f_);
        std::fwrite(&byte_rate, 4, 1, f_);
        std::fwrite(&align, 2, 1, f_);
        std::fwrite(&bits, 2, 1, f_);
        std::fwrite("data", 1, 4, f_);
        std::fwrite(&data_bytes, 4, 1, f_);
        std::fclose(f_);
        f_ = nullptr;
        return true;
    }
    size_t frames() const { return frames_; }

private:
    FILE* f_ = nullptr;
    int sr_ = 24000;
    size_t frames_ = 0;
};

// -------------------------------------------------------------- resampling
static inline std::vector<float> resample_poly(const std::vector<float>& x, int up, int down) {
    if (up == down || x.empty()) return x;
    const int g = std::__gcd(up, down);
    up /= g; down /= g;
    const int max_rate = std::max(up, down);
    const int half = 10 * max_rate;
    const int taps = 2 * half + 1;
    std::vector<double> h(taps);
    {
        const double cutoff = 1.0 / max_rate;
        const double alpha = (taps - 1) / 2.0;
        double sum = 0;
        for (int i = 0; i < taps; ++i) {
            double m = double(i) - alpha;
            double s = (m == 0.0) ? 1.0 : std::sin(M_PI * cutoff * m) / (M_PI * cutoff * m);
            double w = 0.5 * (1.0 + std::cos(M_PI * m / (alpha + 1e-12)));  // Hann-ish
            h[i] = cutoff * s * w;
            sum += h[i];
        }
        for (auto& v : h) v = v / sum * up;
    }
    const long long n_out = (long long)std::ceil((double)x.size() * up / down);
    std::vector<float> y(n_out, 0.f);
    for (long long k = 0; k < n_out; ++k) {
        long long base = half + k * down;
        long long m0 = (base - taps + 1 + up - 1) / up;
        if (m0 < 0) m0 = 0;
        double acc = 0;
        for (int t = 0; t < taps / up + 2; ++t) {
            long long m = m0 + t;
            if (m >= (long long)x.size()) break;
            long long j = base - m * up;
            if (j < 0 || j >= taps) continue;
            acc += h[j] * x[m];
        }
        y[k] = static_cast<float>(acc);
    }
    return y;
}

static inline std::vector<float> resample_to(const std::vector<float>& x, int sr_in, int sr_out) {
    if (sr_in == sr_out) return x;
    return resample_poly(x, sr_out, sr_in);
}

// --------------------------------------------------------------- npy reader
struct Npy {
    std::vector<size_t> shape;
    std::vector<float> data;
};

static inline bool read_npy(const std::string& path, Npy& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "cannot open npy %s\n", path.c_str());
        return false;
    }
    char magic[6];
    if (std::fread(magic, 1, 6, f) != 6 || std::memcmp(magic, "\x93NUMPY", 6)) {
        std::fclose(f);
        return false;
    }
    uint8_t major = 0, minor = 0;
    std::fread(&major, 1, 1, f); std::fread(&minor, 1, 1, f);
    uint32_t hlen = 0;
    if (major == 1) {
        uint16_t h = 0; std::fread(&h, 2, 1, f); hlen = h;
    } else {
        std::fread(&hlen, 4, 1, f);
    }
    std::string header(hlen, '\0');
    std::fread(&header[0], 1, hlen, f);
    std::string descr;
    size_t pos = header.find("descr");
    if (pos != std::string::npos) {
        size_t colon = header.find(':', pos);
        size_t q1 = header.find('\'', colon), q2 = header.find('\'', q1 + 1);
        if (q1 != std::string::npos && q2 != std::string::npos) descr = header.substr(q1 + 1, q2 - q1 - 1);
    }
    std::vector<size_t> shape;
    pos = header.find("shape");
    if (pos != std::string::npos) {
        size_t q1 = header.find('(', pos), q2 = header.find(')', q1);
        std::string s = header.substr(q1 + 1, q2 - q1 - 1);
        size_t i = 0;
        while (i < s.size()) {
            while (i < s.size() && !std::isdigit((unsigned char)s[i])) ++i;
            if (i >= s.size()) break;
            size_t v = 0;
            while (i < s.size() && std::isdigit((unsigned char)s[i])) v = v * 10 + (s[i++] - '0');
            shape.push_back(v);
        }
    }
    size_t count = 1;
    for (auto v : shape) count *= v;
    out.shape = shape;
    out.data.resize(count);
    if (descr == "<f4" || descr == "|f4" || descr == "=f4") {
        std::fread(out.data.data(), 4, count, f);
    } else if (descr == "<f2") {
        std::vector<uint16_t> h(count);
        std::fread(h.data(), 2, count, f);
        for (size_t i = 0; i < count; ++i) {
            uint32_t sign = (h[i] & 0x8000u) << 16;
            uint32_t exp = (h[i] >> 10) & 0x1f;
            uint32_t man = h[i] & 0x3ff;
            uint32_t f32;
            if (exp == 0) {
                f32 = sign | ((man == 0) ? 0u : ((127 - 15 + 1) << 23));
                if (man) {
                    while (!(man & 0x400)) { man <<= 1; }
                    f32 = sign | ((127 - 15) << 23) | ((man & 0x3ff) << 13);
                }
            } else if (exp == 31) {
                f32 = sign | 0x7f800000u | (man << 13);
            } else {
                f32 = sign | ((exp - 15 + 127) << 23) | (man << 13);
            }
            float v;
            std::memcpy(&v, &f32, 4);
            out.data[i] = v;
        }
    } else {
        std::fprintf(stderr, "unsupported npy dtype %s (%s)\n", descr.c_str(), path.c_str());
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    return true;
}

// --------------------------------------------------------------------- FFT
static inline void fft512(std::vector<double>& re, std::vector<double>& im) {
    const int n = 512;
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const double ang = -2.0 * M_PI / len;
        const double wr = std::cos(ang), wi = std::sin(ang);
        for (int i = 0; i < n; i += len) {
            double cwr = 1.0, cwi = 0.0;
            for (int j = 0; j < len / 2; ++j) {
                const int a = i + j, b = i + j + len / 2;
                const double xr = re[b] * cwr - im[b] * cwi;
                const double xi = re[b] * cwi + im[b] * cwr;
                const double nr = cwr * wr - cwi * wi;
                cwi = cwr * wi + cwi * wr;
                cwr = nr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr; im[a] += xi;
            }
        }
    }
}

// log-mel: wav (16 kHz mono) -> [n_mels][frames]
static inline std::vector<float> logmel(const std::vector<float>& wav, const std::vector<float>& mel_fb,
                                        const std::vector<float>& window, int n_mels = 80,
                                        int n_fft = 512, int hop = 160, int win = 400,
                                        float eps = 1e-6f) {
    const int pad = n_fft / 2;
    std::vector<float> x(pad + wav.size() + pad, 0.f);
    // reflect padding
    for (int i = 0; i < pad; ++i) x[pad - 1 - i] = wav[std::min<size_t>(i + 1, wav.size() - 1)];
    for (size_t i = 0; i < wav.size(); ++i) x[pad + i] = wav[i];
    for (int i = 0; i < pad; ++i) {
        size_t src = wav.size() >= 2 ? wav.size() - 2 - i : 0;
        x[pad + wav.size() + i] = wav[std::min(src, wav.size() - 1)];
    }
    const int frames = 1 + static_cast<int>(wav.size()) / hop;
    std::vector<float> mel(static_cast<size_t>(n_mels) * frames, 0.f);
    std::vector<double> re(n_fft), im(n_fft);
    std::vector<double> power(n_fft / 2 + 1);
    for (int t = 0; t < frames; ++t) {
        const int off = t * hop;
        for (int i = 0; i < n_fft; ++i) {
            re[i] = 0.0; im[i] = 0.0;
            if (i >= (n_fft - win) / 2 && i < (n_fft - win) / 2 + win)
                re[i] = x[off + i] * window[i - (n_fft - win) / 2];
        }
        fft512(re, im);
        for (int f = 0; f <= n_fft / 2; ++f) power[f] = re[f] * re[f] + im[f] * im[f];
        for (int m = 0; m < n_mels; ++m) {
            double acc = 0;
            const float* row = &mel_fb[static_cast<size_t>(m) * (n_fft / 2 + 1)];
            for (int f = 0; f <= n_fft / 2; ++f) acc += row[f] * power[f];
            mel[static_cast<size_t>(m) * frames + t] = std::log(std::max((float)acc, eps));
        }
    }
    mel.resize(static_cast<size_t>(n_mels) * frames);
    return mel;  // [n_mels][frames]
}

}  // namespace cutetts
