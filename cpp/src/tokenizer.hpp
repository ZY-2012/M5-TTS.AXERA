#pragma once
// Llama-3 style byte-level BPE from the exported tokenizer.bin (no JSON/SentencePiece).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace m5tts {

inline uint32_t utf8_decode(const std::string& s, size_t& i, uint32_t& cp) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) { cp = c; i += 1; return 1; }
    int n = (c >= 0xF0) ? 3 : (c >= 0xE0) ? 2 : 1;
    if (i + n >= s.size() + 0 && i + n > s.size() - 1) {
        if (i + n > s.size() - 1) n = static_cast<int>(s.size() - i - 1);
    }
    uint32_t v = c & (0xFF >> (n + 1));
    for (int k = 1; k <= n; ++k) v = (v << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
    cp = v;
    i += n + 1;
    return n + 1;
}

inline void utf8_append(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

inline bool is_letter(uint32_t c) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return true;
    return (c >= 0x00C0 && c <= 0x024F) ||  // Latin-1/E extended
           (c >= 0x0370 && c <= 0x03FF) ||  // Greek
           (c >= 0x0400 && c <= 0x052F) ||  // Cyrillic
           (c >= 0x0530 && c <= 0x058F) ||  // Armenian
           (c >= 0x0590 && c <= 0x05FF) ||  // Hebrew
           (c >= 0x0600 && c <= 0x06FF) ||  // Arabic
           (c >= 0x0900 && c <= 0x097F) ||  // Devanagari
           (c >= 0x0E00 && c <= 0x0E7F) ||  // Thai
           (c >= 0x3040 && c <= 0x30FF) ||  // Hiragana / Katakana
           (c >= 0x3400 && c <= 0x4DBF) ||  // CJK ext A
           (c >= 0x4E00 && c <= 0x9FFF) ||  // CJK
           (c >= 0xAC00 && c <= 0xD7AF) ||  // Hangul
           (c >= 0xF900 && c <= 0xFAFF);    // CJK compat
}

inline bool is_number(uint32_t c) {
    if (c >= '0' && c <= '9') return true;
    return (c >= 0x0660 && c <= 0x0669) || (c >= 0x06F0 && c <= 0x06F9) ||
           (c >= 0xFF10 && c <= 0xFF19);
}

inline bool is_space(uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f' ||
           c == 0x00A0 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
           c == 0x202F || c == 0x205F || c == 0x3000;
}

class Tokenizer {
public:
    bool load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
        char magic[8];
        if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "M5TK1", 5) != 0) {
            std::fclose(f);
            std::fprintf(stderr, "bad tokenizer.bin\n");
            return false;
        }
        uint32_t n = 0, m = 0, added = 0;
        read_u32(f, n);
        id_token_.resize(n);
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t len = 0;
            read_u32(f, len);
            id_token_[i].resize(len);
            if (len && std::fread(&id_token_[i][0], 1, len, f) != len) { std::fclose(f); return false; }
            token_id_.emplace(id_token_[i], static_cast<int>(i));
        }
        read_u32(f, m);
        merge_a_.resize(m);
        merge_b_.resize(m);
        merge_out_.resize(m);
        for (uint32_t i = 0; i < m; ++i) {
            uint32_t a, b, out;
            read_u32(f, a); read_u32(f, b); read_u32(f, out);
            merge_a_[i] = a; merge_b_[i] = b; merge_out_[i] = out;
            merge_rank_[key(static_cast<int>(a), static_cast<int>(b))] = static_cast<int>(i);
        }
        read_u32(f, added);
        for (uint32_t i = 0; i < added; ++i) {
            uint32_t tid, len;
            read_u32(f, tid); read_u32(f, len);
            std::string text(len, '\0');
            if (len && std::fread(&text[0], 1, len, f) != len) { std::fclose(f); return false; }
            added_.emplace_back(static_cast<int>(tid), text);
        }
        uint32_t bos = 0, eos = 0, flag = 0;
        read_u32(f, bos); read_u32(f, eos); read_u32(f, flag);
        bos_ = static_cast<int>(bos);
        eos_ = static_cast<int>(eos);
        std::fclose(f);
        std::sort(added_.begin(), added_.end(), [](const auto& x, const auto& y) {
            return x.second.size() > y.second.size();
        });
        build_bytes();
        return valid_;
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> out;
        out.push_back(bos_);
        std::string pending;
        size_t i = 0;
        while (i < text.size()) {
            int hit = -1;
            size_t hit_len = 0;
            if (text[i] == '<' || !added_.empty()) {
                for (const auto& item : added_) {
                    if (item.second.size() <= text.size() - i &&
                        std::memcmp(text.data() + i, item.second.data(), item.second.size()) == 0) {
                        hit = item.first;
                        hit_len = item.second.size();
                        break;
                    }
                }
            }
            if (hit >= 0) {
                if (!pending.empty()) { bpe_text(pending, out); pending.clear(); }
                out.push_back(hit);
                i += hit_len;
            } else {
                pending.push_back(text[i]);
                ++i;
            }
        }
        if (!pending.empty()) bpe_text(pending, out);
        out.push_back(eos_);
        return out;
    }

    int bos() const { return bos_; }
    int eos() const { return eos_; }

private:
    static uint64_t key(int a, int b) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
               static_cast<uint32_t>(b);
    }
    static bool read_u32(FILE* f, uint32_t& v) {
        return std::fread(&v, 4, 1, f) == 1;
    }

    void build_bytes() {
        // GPT-2 byte -> unicode mapping, then to vocab ids
        std::vector<int> bs, cs;
        for (int b = '!'; b <= '~'; ++b) { bs.push_back(b); cs.push_back(b); }
        for (int b = 0xA1; b <= 0xAC; ++b) { bs.push_back(b); cs.push_back(b); }
        for (int b = 0xAE; b <= 0xFF; ++b) { bs.push_back(b); cs.push_back(b); }
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        }
        for (size_t k = 0; k < bs.size(); ++k) {
            std::string mapped;
            utf8_append(mapped, static_cast<uint32_t>(cs[k]));
            auto it = token_id_.find(mapped);
            if (it == token_id_.end()) { valid_ = false; return; }
            byte_id_[bs[k]] = it->second;
        }
    }

    // Split one pretoken (UTF-8) with the llama-3 regex, then BPE each piece.
    void bpe_text(const std::string& text, std::vector<int>& out) const {
        std::vector<uint32_t> cp;
        std::vector<size_t> off;
        size_t i = 0;
        while (i < text.size()) {
            off.push_back(i);
            uint32_t c = 0;
            utf8_decode(text, i, c);
            cp.push_back(c);
        }
        off.push_back(text.size());
        const size_t n = cp.size();
        size_t p = 0;
        while (p < n) {
            size_t q = match(cp, p);
            if (q == p) q = p + 1;
            std::string piece = text.substr(off[p], off[q] - off[p]);
            bpe_piece(piece, out);
            p = q;
        }
    }

    // Returns the end (exclusive) of the longest match at p, or p if none.
    size_t match(const std::vector<uint32_t>& cp, size_t p) const {
        const size_t n = cp.size();
        // 1) (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cp[p] == '\'') {
            auto starts = [&](size_t k, const char* s) {
                for (size_t j = 0; s[j]; ++j)
                    if (k + j >= n || std::tolower(cp[k + j]) != s[j]) return false;
                return true;
            };
            if (starts(p + 1, "re") || starts(p + 1, "ve") || starts(p + 1, "ll")) return p + 3;
            if (starts(p + 1, "s") || starts(p + 1, "t") || starts(p + 1, "m") || starts(p + 1, "d"))
                return p + 2;
        }
        // 2) [^\r\n\p{L}\p{N}]?\p{L}+
        {
            size_t k = p;
            bool optional_ok = !(cp[k] == '\r' || cp[k] == '\n') && !is_letter(cp[k]) && !is_number(cp[k]);
            if (optional_ok && k + 1 < n && is_letter(cp[k + 1])) {
                size_t j = k + 1;
                while (j < n && is_letter(cp[j])) ++j;
                return j;
            }
            if (is_letter(cp[k])) {
                size_t j = k;
                while (j < n && is_letter(cp[j])) ++j;
                return j;
            }
        }
        // 3) \p{N}{1,3}
        if (is_number(cp[p])) {
            size_t j = p;
            while (j < n && is_number(cp[j]) && j - p < 3) ++j;
            return j;
        }
        // 4) ' ?[^\s\p{L}\p{N}]+[\r\n]*'
        {
            size_t k = p;
            if (cp[k] == ' ' && k + 1 < n && !is_space(cp[k + 1]) && !is_letter(cp[k + 1]) && !is_number(cp[k + 1]))
                ++k;
            if (!is_space(cp[k]) && !is_letter(cp[k]) && !is_number(cp[k])) {
                size_t j = k;
                while (j < n && !is_space(cp[j]) && !is_letter(cp[j]) && !is_number(cp[j])) ++j;
                while (j < n && (cp[j] == '\r' || cp[j] == '\n')) ++j;
                return j;
            }
        }
        // 5) \s*[\r\n]+
        {
            size_t j = p;
            while (j < n && is_space(cp[j]) && cp[j] != '\r' && cp[j] != '\n') ++j;
            if (j < n && (cp[j] == '\r' || cp[j] == '\n')) {
                while (j < n && (cp[j] == '\r' || cp[j] == '\n')) ++j;
                return j;
            }
        }
        // 6) \s+(?!\S)
        if (is_space(cp[p])) {
            size_t j = p;
            while (j < n && is_space(cp[j])) ++j;
            if (j == n) return j;
            if (j - p >= 2) return j - 1;   // leave one space for the next token
            return p;
        }
        // 7) \s+
        if (is_space(cp[p])) {
            size_t j = p;
            while (j < n && is_space(cp[j])) ++j;
            return j;
        }
        return p;
    }

    void bpe_piece(const std::string& piece, std::vector<int>& out) const {
        std::vector<int> ids;
        ids.reserve(piece.size());
        for (unsigned char c : piece) ids.push_back(byte_id_[c]);
        while (ids.size() > 1) {
            int best = INT32_MAX;
            size_t at = 0;
            for (size_t i = 0; i + 1 < ids.size(); ++i) {
                auto it = merge_rank_.find(key(ids[i], ids[i + 1]));
                if (it != merge_rank_.end() && it->second < best) { best = it->second; at = i; }
            }
            if (best == INT32_MAX) break;
            std::vector<int> next;
            next.reserve(ids.size());
            for (size_t i = 0; i < ids.size();) {
                if (i + 1 < ids.size()) {
                    auto it = merge_rank_.find(key(ids[i], ids[i + 1]));
                    if (it != merge_rank_.end() && it->second == best) {
                        next.push_back(merge_out_[best]);
                        i += 2;
                        continue;
                    }
                }
                next.push_back(ids[i]);
                ++i;
            }
            ids.swap(next);
        }
        out.insert(out.end(), ids.begin(), ids.end());
    }

    std::vector<std::string> id_token_;
    std::unordered_map<std::string, int> token_id_;
    std::vector<uint32_t> merge_a_, merge_b_, merge_out_;
    std::unordered_map<uint64_t, int> merge_rank_;
    std::vector<std::pair<int, std::string>> added_;
    int byte_id_[256] = {0};
    int bos_ = 128000, eos_ = 128001;
    bool valid_ = true;
};

}  // namespace m5tts
