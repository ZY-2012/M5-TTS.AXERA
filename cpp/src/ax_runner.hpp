#pragma once
// Minimal AX650 AX_ENGINE wrapper: N float32 inputs / M float32 outputs, name based.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "ax_engine_api.h"
#include "ax_engine_type.h"
#include "ax_sys_api.h"

namespace cutetts {

inline bool ax_runtime_init() {
    static bool done = false;
    if (done) return true;
    if (AX_SYS_Init() != 0) {
        std::fprintf(stderr, "AX_SYS_Init failed\n");
        return false;
    }
    AX_ENGINE_NPU_ATTR_T attr;
    std::memset(&attr, 0, sizeof(attr));
    attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    if (AX_ENGINE_Init(&attr) != 0) {
        std::fprintf(stderr, "AX_ENGINE_Init failed\n");
        return false;
    }
    done = true;
    return true;
}

class AxModel {
public:
    AxModel() = default;
    ~AxModel() { unload(); }
    AxModel(const AxModel&) = delete;
    AxModel& operator=(const AxModel&) = delete;

    bool load(const std::string& path) {
        if (!ax_runtime_init()) return false;
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "cannot open model %s\n", path.c_str());
            return false;
        }
        f.seekg(0, std::ios::end);
        const size_t n = static_cast<size_t>(f.tellg());
        f.seekg(0, std::ios::beg);
        data_.resize(n);
        f.read(data_.data(), static_cast<std::streamsize>(n));

        AX_ENGINE_HANDLE_EXTRA_T extra;
        std::memset(&extra, 0, sizeof(extra));
        extra.pName = const_cast<AX_S8*>(reinterpret_cast<const AX_S8*>("cutetts"));
        if (AX_ENGINE_CreateHandleV2(&handle_, data_.data(), static_cast<AX_U32>(n), &extra) != 0 ||
            !handle_) {
            std::fprintf(stderr, "AX_ENGINE_CreateHandleV2 failed: %s\n", path.c_str());
            return false;
        }
        if (AX_ENGINE_CreateContextV2(handle_, &context_) != 0 || !context_) {
            std::fprintf(stderr, "AX_ENGINE_CreateContextV2 failed: %s\n", path.c_str());
            return false;
        }
        if (AX_ENGINE_GetIOInfo(handle_, &info_) != 0 || !info_) {
            std::fprintf(stderr, "AX_ENGINE_GetIOInfo failed: %s\n", path.c_str());
            return false;
        }
        io_ = new AX_ENGINE_IO_T;
        std::memset(io_, 0, sizeof(*io_));
        io_->pInputs = new AX_ENGINE_IO_BUFFER_T[info_->nInputSize];
        io_->pOutputs = new AX_ENGINE_IO_BUFFER_T[info_->nOutputSize];
        std::memset(io_->pInputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info_->nInputSize);
        std::memset(io_->pOutputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info_->nOutputSize);
        io_->nInputSize = info_->nInputSize;
        io_->nOutputSize = info_->nOutputSize;
        for (AX_U32 i = 0; i < info_->nInputSize; ++i) {
            auto& b = io_->pInputs[i];
            b.nSize = info_->pInputs[i].nSize;
            if (AX_SYS_MemAllocCached(reinterpret_cast<AX_U64*>(&b.phyAddr), &b.pVirAddr, b.nSize,
                                      128, reinterpret_cast<const AX_S8*>("cutetts")) != 0)
                return false;
            std::memset(b.pVirAddr, 0, b.nSize);
            in_index_[info_->pInputs[i].pName] = static_cast<int>(i);
        }
        for (AX_U32 i = 0; i < info_->nOutputSize; ++i) {
            auto& b = io_->pOutputs[i];
            b.nSize = info_->pOutputs[i].nSize;
            if (AX_SYS_MemAllocCached(reinterpret_cast<AX_U64*>(&b.phyAddr), &b.pVirAddr, b.nSize,
                                      128, reinterpret_cast<const AX_S8*>("cutetts")) != 0)
                return false;
            out_index_[info_->pOutputs[i].pName] = static_cast<int>(i);
        }
        return true;
    }

    int input_index(const std::string& name) const {
        auto it = in_index_.find(name);
        return it == in_index_.end() ? -1 : it->second;
    }
    int output_index(const std::string& name) const {
        auto it = out_index_.find(name);
        return it == out_index_.end() ? -1 : it->second;
    }
    size_t input_bytes(const std::string& name) const {
        int i = input_index(name);
        return i < 0 ? 0 : info_->pInputs[i].nSize;
    }
    size_t output_bytes(const std::string& name) const {
        int i = output_index(name);
        return i < 0 ? 0 : info_->pOutputs[i].nSize;
    }

    bool set(const std::string& name, const float* data, size_t floats) {
        const int idx = input_index(name);
        if (idx < 0) {
            std::fprintf(stderr, "no input '%s'\n", name.c_str());
            return false;
        }
        auto& b = io_->pInputs[idx];
        if (floats * sizeof(float) > b.nSize) {
            std::fprintf(stderr, "input '%s' too large (%zu > %u)\n", name.c_str(),
                         floats * sizeof(float), b.nSize);
            return false;
        }
        std::memcpy(b.pVirAddr, data, floats * sizeof(float));
        AX_SYS_MflushCache(b.phyAddr, b.pVirAddr, b.nSize);
        return true;
    }
    template <typename T>
    bool set(const std::string& name, const std::vector<T>& v) {
        return set(name, v.data(), v.size());
    }

    // warm up the NPU: one dummy inference with zero inputs (excluded from TTFT)
    bool warmup() {
        for (AX_U32 i = 0; i < info_->nInputSize; ++i)
            std::memset(io_->pInputs[i].pVirAddr, 0, io_->pInputs[i].nSize);
        return run();
    }

    bool loaded() const { return handle_ != nullptr; }

    bool run() {
        if (AX_ENGINE_RunSyncV2(handle_, context_, io_) != 0) return false;
        for (AX_U32 i = 0; i < info_->nOutputSize; ++i) {
            auto& b = io_->pOutputs[i];
            AX_SYS_MinvalidateCache(b.phyAddr, b.pVirAddr, b.nSize);
        }
        return true;
    }

    bool get(const std::string& name, float* out, size_t floats) const {
        const int idx = output_index(name);
        if (idx < 0) {
            std::fprintf(stderr, "no output '%s'\n", name.c_str());
            return false;
        }
        auto& b = io_->pOutputs[idx];
        if (floats * sizeof(float) > b.nSize) {
            std::fprintf(stderr, "output '%s' expected %zu floats, buffer %u bytes\n", name.c_str(),
                         floats, b.nSize);
            return false;
        }
        std::memcpy(out, b.pVirAddr, floats * sizeof(float));
        return true;
    }
    std::vector<float> get(const std::string& name, size_t floats) const {
        std::vector<float> v(floats, 0.f);
        if (!get(name, v.data(), floats)) v.clear();
        return v;
    }

private:
    void unload() {
        if (io_) {
            for (AX_U32 i = 0; i < io_->nInputSize; ++i)
                if (io_->pInputs[i].pVirAddr) AX_SYS_MemFree(io_->pInputs[i].phyAddr, io_->pInputs[i].pVirAddr);
            for (AX_U32 i = 0; i < io_->nOutputSize; ++i)
                if (io_->pOutputs[i].pVirAddr) AX_SYS_MemFree(io_->pOutputs[i].phyAddr, io_->pOutputs[i].pVirAddr);
            delete[] io_->pInputs;
            delete[] io_->pOutputs;
            delete io_;
            io_ = nullptr;
        }
        if (handle_) { AX_ENGINE_DestroyHandle(handle_); handle_ = nullptr; }
    }

    void* handle_ = nullptr;
    void* context_ = nullptr;
    AX_ENGINE_IO_INFO_T* info_ = nullptr;
    AX_ENGINE_IO_T* io_ = nullptr;
    std::vector<char> data_;
    std::map<std::string, int> in_index_;
    std::map<std::string, int> out_index_;
};

}  // namespace cutetts
