// Glow Profond: Metal implementation of glow_profond::glow_layer.
//
// Same pipeline and parameters as the CPU core (core/src/glow.cpp): soft
// threshold, optional global downsample, pyramid of truncating 2x
// downsamples, variance-compensated separable Gaussians, tint and weight
// per level, tent cascade back up, exposure folded into the weights, then
// saturation. Depth (spread, occlusion) and lens dirt are CPU-only for now:
// check supports() and fall back to the core when it says no.
// Plain C++ interface (no Objective-C types) so the AE wrapper can include it.
#pragma once

#include <memory>
#include <string>

#include "glow_profond/glow.hpp"

namespace glow_profond {
namespace metal {

struct Timing {
    double gpu_ms = 0.0;    // command buffer execution on the GPU
    double total_ms = 0.0;  // including upload, encoding and readback
};

class GlowMetal {
public:
    // Null (with *error set) if there is no Metal device or the kernels
    // fail to compile.
    static std::unique_ptr<GlowMetal> create(std::string* error);
    // Same, on a given MTLDevice (the host's, e.g. After Effects' GPU device).
    static std::unique_ptr<GlowMetal> create_for_device(void* mtl_device, std::string* error);
    ~GlowMetal();

    // Whether this path renders p exactly like glow_layer() (with the given
    // optional inputs present or not).
    static bool supports(const GlowParams& p, bool has_depth, bool has_dirt);

    // CPU image in (3 or 4 channels, linear, premultiplied), 3-channel glow
    // out, like glow_layer(src_lin, p). For tests and tools.
    bool glow_layer(const Image& src_lin, const GlowParams& p, Image& out, Timing* timing = nullptr);

    // Encodes the glow into an existing MTLCommandBuffer. src and dst are
    // MTLBuffers of width * height float4 (RGB used, A ignored / zeroed).
    // For the After Effects GPU path.
    bool encode_glow_layer(void* command_buffer, void* src_buffer, void* dst_buffer, int width, int height,
                           const GlowParams& p);

    // A host frame: MTLBuffer of BGRA float4 premultiplied pixels, rows
    // `pitch` pixels apart, placed at (dx, dy) in the work area.
    struct HostFrame {
        void* buffer = nullptr;
        int width = 0, height = 0, pitch = 0;
        int dx = 0, dy = 0;
    };

    // The whole render_premultiplied() on the GPU (no depth, dirt or
    // neighbours: see supports()): src is read into a work area of
    // work_width x work_height, the glow computed there, and dst written
    // from it. clamp_premult: 8/16-bit project, cap colour at alpha.
    bool encode_render_premultiplied(void* command_buffer, const HostFrame& src, const HostFrame& dst,
                                     int work_width, int work_height, const GlowParams& p, bool clamp_premult);

    // encode_render_premultiplied() in a new command buffer of the host's
    // MTLCommandQueue, committed (not waited on: the host orders its queue).
    bool submit_render_premultiplied(void* command_queue, const HostFrame& src, const HostFrame& dst,
                                     int work_width, int work_height, const GlowParams& p, bool clamp_premult);

    // Test helper: render_premultiplied(rgba, p) through the GPU path.
    bool render_premultiplied(const Image& rgba, const GlowParams& p, Image& out, Timing* timing = nullptr);

private:
    GlowMetal();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace metal
}  // namespace glow_profond
