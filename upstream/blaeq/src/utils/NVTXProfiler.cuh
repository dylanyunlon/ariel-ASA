//
// Created by cuda01 on 2026/1/9.
//

#ifndef BLAEQ_CUDA_NVTXPROFILER_CUH
#define BLAEQ_CUDA_NVTXPROFILER_CUH

#include <nvtx3/nvToolsExt.h>
#include <string>
#include <unordered_map>
#include <atomic>
#include <random>
#include <array>

namespace NvtxProfilerColor {
    enum Color : uint32_t {
        Red         = 0xFF0000FF,
        Green       = 0xFF00FF00,
        Blue        = 0xFFFF0000,
        Cyan        = 0xFFFFFF00,
        Magenta     = 0xFFFF00FF,
        Yellow      = 0xFF00FFFF,
        Orange      = 0xFFFF8000,
        Purple      = 0xFF8000FF,
        SpringGreen = 0xFF00FF80,
        LimeGreen   = 0xFF80FF00,
        Rose        = 0xFFFF0080,
        SkyBlue     = 0xFF0080FF
    };
}

class NvtxProfiler {
public:
    enum class ColorMode {
        None,
        Fixed,
        Random
    };

private:
    bool active_ = false;

    static std::unordered_map<std::string, std::atomic<uint32_t>>& getCounters() {
        static std::unordered_map<std::string, std::atomic<uint32_t>> counters;
        return counters;
    }

    static constexpr std::array<uint32_t, 12> COLOR_PALETTE = {
        0xFF0000FF,
        0xFF00FF00,
        0xFFFF0000,
        0xFFFFFF00,
        0xFFFF00FF,
        0xFF00FFFF,
        0xFFFF8000,
        0xFF8000FF,
        0xFF00FF80,
        0xFF80FF00,
        0xFFFF0080,
        0xFF0080FF
    };

    static uint32_t getRandomColor() {
        thread_local std::mt19937 gen(std::random_device{}());
        thread_local std::uniform_int_distribution<size_t> dist(0, COLOR_PALETTE.size() - 1);
        return COLOR_PALETTE[dist(gen)];
    }

public:
    explicit NvtxProfiler(
        const char* name,
        const ColorMode colorMode = ColorMode::None,
        const uint32_t fixedColor = 0xFFFFFFFF
    ) : active_(true) {

        const auto count = ++getCounters()[name];

        char buffer[256];
        if (count > 1) {
            snprintf(buffer, sizeof(buffer), "%s#%u", name, count);
        } else {
            snprintf(buffer, sizeof(buffer), "%s", name);
        }

        nvtxEventAttributes_t attrib = {0};
        attrib.version = NVTX_VERSION;
        attrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attrib.message.ascii = buffer;

        if (colorMode != ColorMode::None) {
            attrib.colorType = NVTX_COLOR_ARGB;
            attrib.color = (colorMode == ColorMode::Random)
                ? getRandomColor()
                : fixedColor;
        }

        nvtxRangePushEx(&attrib);
    }

    NvtxProfiler(const NvtxProfiler&) = delete;
    NvtxProfiler& operator=(const NvtxProfiler&) = delete;

    NvtxProfiler(NvtxProfiler&& other) noexcept : active_(other.active_) {
        other.active_ = false;
    }

    NvtxProfiler& operator=(NvtxProfiler&& other) noexcept {
        if (this != &other) {
            release();
            active_ = other.active_;
            other.active_ = false;
        }
        return *this;
    }

    void release() {
        if (active_) {
            nvtxRangePop();
            active_ = false;
        }
    }

    ~NvtxProfiler() {
        release();
    }

    bool isActive() const { return active_; }
};

#define NVTX_PROFILE(name) \
    NvtxProfiler nvtx_profiler_##__LINE__(name)

#define NVTX_PROFILE_COLOR(name, color) \
    NvtxProfiler nvtx_profiler_##__LINE__(name, NvtxProfiler::ColorMode::Fixed, color)

#define NVTX_PROFILE_RANDOM(name) \
    NvtxProfiler nvtx_profiler_##__LINE__(name, NvtxProfiler::ColorMode::Random)

#endif //BLAEQ_CUDA_NVTXPROFILER_CUH