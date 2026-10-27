// Glow Profond: Metal host code. Builds the same level plan as
// glow_layer() in core/src/glow.cpp and encodes one kernel per CPU step.
#import <Metal/Metal.h>

#include "GlowMetal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

namespace glow_profond {
namespace metal {
namespace {

const char kKernelSource[] =
#include "GlowKernels.metal.inc"
    ;

// Mirrors of private helpers in core/src/glow.cpp; keep them in step.
constexpr float kTentStepVariance = 0.6875f;

std::vector<float> gauss_kernel(float sigma) {
    const int r = std::max(1, int(std::ceil(3.0f * sigma)));
    const float s = std::max(sigma, 1e-3f);
    std::vector<float> k(static_cast<size_t>(2 * r + 1));
    float sum = 0.0f;
    for (int i = -r; i <= r; ++i) {
        const float x = float(i) / s;
        k[size_t(i + r)] = std::exp(-0.5f * x * x);
        sum += k[size_t(i + r)];
    }
    for (float& v : k) v /= sum;
    return k;
}

Rgb tint_for_level(const std::vector<Rgb>& tint, int k, int n) {
    if (tint.empty()) return {1.0f, 1.0f, 1.0f};
    if (tint.size() == 1) return tint[0];
    const float t = float(k) / float(std::max(n - 1, 1)) * float(tint.size() - 1);
    const int i = std::min(int(t), int(tint.size()) - 2);
    const float f = t - float(i);
    Rgb c;
    for (int ch = 0; ch < 3; ++ch) c[ch] = tint[i][ch] * (1.0f - f) + tint[i + 1][ch] * f;
    return c;
}

// Layouts shared with GlowKernels.metal.
struct Dims {
    uint32_t w, h;
};
struct BlurParams {
    int32_t r[4];
    int32_t off[4];
    float gain[4];
};
struct FinishParams {
    Dims acc, out;
    float saturation;
    int32_t use_saturation;
};
struct PrepareParams {
    uint32_t src_w, src_h, src_pitch;
    int32_t dx, dy;
    uint32_t work_w, work_h;
    int32_t unmult, gamma, clamp_premult;
};
struct CompositeParams {
    uint32_t dst_w, dst_h, dst_pitch;
    int32_t dx, dy;
    uint32_t work_w;
    int32_t view, screen, gamma, tonemap;
};
struct TentTaps {
    int32_t idx[4];
    float w[4];
    int32_t n;
    int32_t pad[3];
};
static_assert(sizeof(TentTaps) == 48, "TentTaps layout must match the Metal struct");

std::vector<TentTaps> tent_taps(int target, int n) {
    static const float kTent[3] = {0.25f, 0.5f, 0.25f};
    std::vector<TentTaps> t(static_cast<size_t>(target));
    for (int i = 0; i < target; ++i) {
        TentTaps& tt = t[size_t(i)];
        std::memset(&tt, 0, sizeof(tt));
        auto add = [&](int idx, float w) {
            for (int m = 0; m < tt.n; ++m) {
                if (tt.idx[m] == idx) {
                    tt.w[m] += w;
                    return;
                }
            }
            tt.idx[tt.n] = idx;
            tt.w[tt.n++] = w;
        };
        const float base = (float(i) + 0.5f) * float(n) / float(target) - 0.5f;
        for (int d = 0; d < 3; ++d) {
            const float c = std::clamp(base + float(d - 1), 0.0f, float(n - 1));
            const int i0 = int(std::floor(c));
            const float f = c - float(i0);
            add(i0, kTent[d] * (1.0f - f));
            add(std::min(i0 + 1, n - 1), kTent[d] * f);
        }
    }
    return t;
}

// Everything glow_layer() decides before touching pixels.
struct Plan {
    struct Level {
        int w = 0, h = 0;
        std::vector<float> kx, ky;  // three channels back to back
        BlurParams bx{}, by{};
    };
    float threshold = 0.0f, knee = 0.0f;
    std::vector<Dims> chain;  // full size, then each global downsample
    std::vector<Level> levels;
};

Plan make_plan(int W, int H, const GlowParams& p) {
    Plan plan;
    plan.threshold = p.threshold;
    plan.knee = std::max(p.threshold * p.threshold_smooth, 1e-5f);
    plan.chain.push_back({uint32_t(W), uint32_t(H)});
    const int extra = int(std::log2(float(std::max(p.downsample, 1))));
    for (int i = 0; i < extra; ++i) {
        const Dims& d = plan.chain.back();
        plan.chain.push_back({d.w / 2, d.h / 2});
    }
    const Dims base = plan.chain.back();
    if (base.w == 0 || base.h == 0) return plan;

    const float scale = float(H) / float(base.h);
    // The core's level plan (per-level sigma, unnormalized weights), so the
    // GPU follows any change to it.
    const LevelPlan lp = level_plan(p);
    const int n = int(lp.sigmas.size());
    std::vector<float> weights = lp.weights;
    const float exposure_gain = std::pow(2.0f, p.exposure);
    for (float& w : weights) w *= exposure_gain;

    const float ax = std::sqrt(p.aspect), ay = 1.0f / std::sqrt(p.aspect);
    int w = int(base.w), h = int(base.h);
    for (int k = 0; k < n; ++k) {
        if (k > 0) {
            w /= 2;
            h /= 2;
            if (w == 0 || h == 0) break;  // tiny images: fewer levels
        }
        Plan::Level lv;
        lv.w = w;
        lv.h = h;
        // Shrunk by the variance the upsample chain adds back (see glow_layer).
        const float extra_var = kTentStepVariance * (4.0f / 3.0f) * (1.0f - std::pow(4.0f, -float(k)));
        const Rgb tint = tint_for_level(p.tint, k, n);
        // Sigma in this level's own pixels.
        const float sigma_level = lp.sigmas[size_t(k)] / (std::pow(2.0f, float(k)) * scale);
        for (int c = 0; c < 3; ++c) {
            const float s = sigma_level * p.rgb_radius[c];
            const float sx = std::sqrt(std::max((s * ax) * (s * ax) - extra_var, 0.0f));
            const float sy = std::sqrt(std::max((s * ay) * (s * ay) - extra_var, 0.0f));
            auto push = [](std::vector<float>& dst, BlurParams& bp, int c, float sigma) {
                bp.off[c] = int(dst.size());
                if (sigma >= 0.3f) {
                    const std::vector<float> k = gauss_kernel(sigma);
                    bp.r[c] = int(k.size()) / 2;
                    dst.insert(dst.end(), k.begin(), k.end());
                } else {
                    bp.r[c] = 0;  // the CPU skips this axis: identity
                    dst.push_back(1.0f);
                }
            };
            push(lv.kx, lv.bx, c, sx);
            push(lv.ky, lv.by, c, sy);
            lv.bx.gain[c] = 1.0f;
            lv.by.gain[c] = weights[k] * tint[c];
        }
        plan.levels.push_back(std::move(lv));
    }
    return plan;
}

}  // namespace

struct GlowMetal::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> threshold, down, blur_h, blur_v, tent_x, tent_y_add, finish, prepare, composite;

    id<MTLBuffer> buffer(size_t pixels) {
        return [device newBufferWithLength:std::max<size_t>(pixels, 1) * 16 options:MTLResourceStorageModePrivate];
    }

    template <class T>
    static void bytes(id<MTLComputeCommandEncoder> enc, const T& v, NSUInteger index) {
        [enc setBytes:&v length:sizeof(T) atIndex:index];
    }

    void floats(id<MTLComputeCommandEncoder> enc, const std::vector<float>& v, NSUInteger index) {
        const size_t len = v.size() * sizeof(float);
        if (len <= 4096) {
            [enc setBytes:v.data() length:len atIndex:index];  // small constant data
        } else {
            [enc setBuffer:[device newBufferWithBytes:v.data() length:len options:MTLResourceStorageModeShared]
                    offset:0
                   atIndex:index];
        }
    }

    static void dispatch(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps, uint32_t w, uint32_t h) {
        [enc setComputePipelineState:ps];
        const NSUInteger tw = ps.threadExecutionWidth;
        const NSUInteger th = std::max<NSUInteger>(1, ps.maxTotalThreadsPerThreadgroup / tw);
        [enc dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:MTLSizeMake(tw, std::min<NSUInteger>(th, 16), 1)];
    }

    // Encodes the whole pipeline: src (W x H float4) -> dst (W x H float4).
    bool encode(id<MTLCommandBuffer> cb, id<MTLBuffer> src, id<MTLBuffer> dst, int W, int H, const GlowParams& p) {
        const Plan plan = make_plan(W, H, p);
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) return false;
        if (plan.levels.empty()) {
            // Image too small to glow: black, like glow_layer().
            [enc endEncoding];
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            [blit fillBuffer:dst range:NSMakeRange(0, size_t(W) * H * 16) value:0];
            [blit endEncoding];
            return true;
        }

        // Threshold, then the optional global downsample chain.
        const Dims full = plan.chain.front();
        id<MTLBuffer> cur = buffer(size_t(W) * H);
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:cur offset:0 atIndex:1];
        bytes(enc, full, 2);
        const float tk[2] = {plan.threshold, plan.knee};
        [enc setBytes:tk length:sizeof(tk) atIndex:3];
        dispatch(enc, threshold, full.w, full.h);
        for (size_t i = 1; i < plan.chain.size(); ++i) {
            id<MTLBuffer> next = buffer(size_t(plan.chain[i].w) * plan.chain[i].h);
            [enc setBuffer:cur offset:0 atIndex:0];
            [enc setBuffer:next offset:0 atIndex:1];
            bytes(enc, plan.chain[i - 1], 2);
            bytes(enc, plan.chain[i], 3);
            dispatch(enc, down, plan.chain[i].w, plan.chain[i].h);
            cur = next;
        }

        // Pyramid.
        const size_t n = plan.levels.size();
        std::vector<id<MTLBuffer>> lv(n);
        lv[0] = cur;
        for (size_t k = 1; k < n; ++k) {
            const Dims sd{uint32_t(plan.levels[k - 1].w), uint32_t(plan.levels[k - 1].h)};
            const Dims dd{uint32_t(plan.levels[k].w), uint32_t(plan.levels[k].h)};
            lv[k] = buffer(size_t(dd.w) * dd.h);
            [enc setBuffer:lv[k - 1] offset:0 atIndex:0];
            [enc setBuffer:lv[k] offset:0 atIndex:1];
            bytes(enc, sd, 2);
            bytes(enc, dd, 3);
            dispatch(enc, down, dd.w, dd.h);
        }

        // Blur, weight and tint every level in place (via one scratch).
        id<MTLBuffer> scratch = buffer(size_t(plan.levels[0].w) * plan.levels[0].h);
        for (size_t k = 0; k < n; ++k) {
            const Plan::Level& L = plan.levels[k];
            const Dims d{uint32_t(L.w), uint32_t(L.h)};
            [enc setBuffer:lv[k] offset:0 atIndex:0];
            [enc setBuffer:scratch offset:0 atIndex:1];
            bytes(enc, d, 2);
            floats(enc, L.kx, 3);
            bytes(enc, L.bx, 4);
            dispatch(enc, blur_h, d.w, d.h);
            [enc setBuffer:scratch offset:0 atIndex:0];
            [enc setBuffer:lv[k] offset:0 atIndex:1];
            floats(enc, L.ky, 3);
            bytes(enc, L.by, 4);
            dispatch(enc, blur_v, d.w, d.h);
        }

        // Tent cascade, coarsest to finest: lv[k] += tent(lv[k + 1]).
        id<MTLBuffer> acc = lv[n - 1];
        for (int k = int(n) - 2; k >= 0; --k) {
            const Dims sd{uint32_t(plan.levels[k + 1].w), uint32_t(plan.levels[k + 1].h)};
            const Dims dd{uint32_t(plan.levels[k].w), uint32_t(plan.levels[k].h)};
            const Dims rows{dd.w, sd.h};
            const std::vector<TentTaps> tx = tent_taps(int(dd.w), int(sd.w));
            const std::vector<TentTaps> ty = tent_taps(int(dd.h), int(sd.h));
            id<MTLBuffer> txb = [device newBufferWithBytes:tx.data() length:tx.size() * sizeof(TentTaps)
                                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> tyb = [device newBufferWithBytes:ty.data() length:ty.size() * sizeof(TentTaps)
                                                   options:MTLResourceStorageModeShared];
            // scratch holds dd.w x sd.h <= level 0 size.
            [enc setBuffer:acc offset:0 atIndex:0];
            [enc setBuffer:scratch offset:0 atIndex:1];
            bytes(enc, sd, 2);
            bytes(enc, rows, 3);
            [enc setBuffer:txb offset:0 atIndex:4];
            dispatch(enc, tent_x, rows.w, rows.h);
            id<MTLBuffer> out = buffer(size_t(dd.w) * dd.h);
            [enc setBuffer:scratch offset:0 atIndex:0];
            [enc setBuffer:out offset:0 atIndex:1];
            bytes(enc, dd, 2);
            [enc setBuffer:lv[k] offset:0 atIndex:3];
            [enc setBuffer:tyb offset:0 atIndex:4];
            dispatch(enc, tent_y_add, dd.w, dd.h);
            acc = out;
        }

        // Full size, saturation.
        FinishParams fp{};
        fp.acc = {uint32_t(plan.levels[0].w), uint32_t(plan.levels[0].h)};
        fp.out = full;
        fp.saturation = p.saturation;
        fp.use_saturation = p.saturation != 1.0f;
        [enc setBuffer:acc offset:0 atIndex:0];
        [enc setBuffer:dst offset:0 atIndex:1];
        bytes(enc, fp, 2);
        dispatch(enc, finish, full.w, full.h);
        [enc endEncoding];
        return true;
    }
};

GlowMetal::GlowMetal() : impl_(new Impl) {}
GlowMetal::~GlowMetal() = default;

std::unique_ptr<GlowMetal> GlowMetal::create(std::string* error) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            if (error) *error = "no Metal device";
            return nullptr;
        }
        return create_for_device((__bridge void*)device, error);
    }
}

std::unique_ptr<GlowMetal> GlowMetal::create_for_device(void* mtl_device, std::string* error) {
    @autoreleasepool {
        std::unique_ptr<GlowMetal> g(new GlowMetal);
        Impl& m = *g->impl_;
        m.device = (__bridge id<MTLDevice>)mtl_device;
        if (!m.device) {
            if (error) *error = "no Metal device";
            return nullptr;
        }
        MTLCompileOptions* opts = [MTLCompileOptions new];
        // IEEE behaviour, to match the CPU bit for bit.
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSError* err = nil;
        id<MTLLibrary> lib = [m.device newLibraryWithSource:@(kKernelSource) options:opts error:&err];
        if (!lib) {
            if (error) *error = std::string("Metal compile failed: ") + err.localizedDescription.UTF8String;
            return nullptr;
        }
        auto pipeline = [&](const char* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> f = [lib newFunctionWithName:@(name)];
            NSError* e = nil;
            id<MTLComputePipelineState> ps = f ? [m.device newComputePipelineStateWithFunction:f error:&e] : nil;
            if (!ps && error) *error = std::string("no pipeline for ") + name;
            return ps;
        };
        m.threshold = pipeline("soft_threshold");
        m.down = pipeline("downsample2");
        m.blur_h = pipeline("blur_h");
        m.blur_v = pipeline("blur_v");
        m.tent_x = pipeline("tent_x");
        m.tent_y_add = pipeline("tent_y_add");
        m.finish = pipeline("finish");
        m.prepare = pipeline("prepare");
        m.composite = pipeline("composite");
        if (!m.threshold || !m.down || !m.blur_h || !m.blur_v || !m.tent_x || !m.tent_y_add || !m.finish ||
            !m.prepare || !m.composite) {
            return nullptr;
        }
        m.queue = [m.device newCommandQueue];
        return g;
    }
}

bool GlowMetal::supports(const GlowParams& p, bool has_depth, bool has_dirt) {
    const bool depth_used = has_depth && (p.depth_spread > 0.0f || p.occlusion);
    const bool dirt_used = has_dirt && p.dirt_amount > 0.0f;
    return !depth_used && !dirt_used;
}

bool GlowMetal::glow_layer(const Image& src, const GlowParams& p, Image& out, Timing* timing) {
    @autoreleasepool {
        const auto t0 = std::chrono::steady_clock::now();
        const int W = src.width, H = src.height;
        out = Image(W, H, 3);
        if (W == 0 || H == 0) return true;
        const size_t px = size_t(W) * H;
        id<MTLBuffer> in = [impl_->device newBufferWithLength:px * 16 options:MTLResourceStorageModeShared];
        id<MTLBuffer> res = [impl_->device newBufferWithLength:px * 16 options:MTLResourceStorageModeShared];
        auto upload = [&](const Image& img, id<MTLBuffer> buf) {
            float* f = static_cast<float*>(buf.contents);
            parallel_for_rows(H, [&](int y) {
                for (int x = 0; x < W; ++x) {
                    const float* s = img.px(x, y);
                    float* d = f + (size_t(y) * W + x) * 4;
                    d[0] = s[0];
                    d[1] = s[1];
                    d[2] = s[2];
                    d[3] = 0.0f;
                }
            });
        };
        upload(src, in);
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        if (!impl_->encode(cb, in, res, W, H, p)) return false;
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return false;
        const float* r = static_cast<const float*>(res.contents);
        parallel_for_rows(H, [&](int y) {
            for (int x = 0; x < W; ++x) {
                const float* s = r + (size_t(y) * W + x) * 4;
                float* d = out.px(x, y);
                d[0] = s[0];
                d[1] = s[1];
                d[2] = s[2];
            }
        });
        if (timing) {
            timing->gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            timing->total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
        return true;
    }
}

bool GlowMetal::encode_glow_layer(void* command_buffer, void* src_buffer, void* dst_buffer, int width, int height,
                                  const GlowParams& p) {
    return impl_->encode((__bridge id<MTLCommandBuffer>)command_buffer, (__bridge id<MTLBuffer>)src_buffer,
                         (__bridge id<MTLBuffer>)dst_buffer, width, height, p);
}

bool GlowMetal::encode_render_premultiplied(void* command_buffer, const HostFrame& src, const HostFrame& dst,
                                            int work_width, int work_height, const GlowParams& p, bool clamp_premult) {
    Impl& m = *impl_;
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)command_buffer;
    const size_t px = size_t(work_width) * size_t(work_height);
    if (px == 0) return true;
    id<MTLBuffer> work = m.buffer(px);
    {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) return false;
        PrepareParams pp{uint32_t(src.width), uint32_t(src.height), uint32_t(src.pitch), src.dx, src.dy,
                         uint32_t(work_width), uint32_t(work_height), p.unmult ? 1 : 0, p.gamma_correct ? 1 : 0,
                         clamp_premult ? 1 : 0};
        [enc setBuffer:(__bridge id<MTLBuffer>)src.buffer offset:0 atIndex:0];
        [enc setBuffer:work offset:0 atIndex:1];
        Impl::bytes(enc, pp, 2);
        Impl::dispatch(enc, m.prepare, uint32_t(work_width), uint32_t(work_height));
        [enc endEncoding];
    }
    id<MTLBuffer> glow = work;  // unused by the Source view
    if (p.view != View::Source) {
        glow = m.buffer(px);
        if (!m.encode(cb, work, glow, work_width, work_height, p)) return false;
    }
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) return false;
    const int view = p.view == View::Glow ? 1 : p.view == View::Source ? 2 : p.view == View::Behind ? 3 : 0;
    CompositeParams cp{uint32_t(dst.width), uint32_t(dst.height), uint32_t(dst.pitch), dst.dx, dst.dy,
                       uint32_t(work_width), view, p.blend == Blend::Screen ? 1 : 0, p.gamma_correct ? 1 : 0,
                       p.tonemap ? 1 : 0};
    [enc setBuffer:work offset:0 atIndex:0];
    [enc setBuffer:glow offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)dst.buffer offset:0 atIndex:2];
    Impl::bytes(enc, cp, 3);
    Impl::dispatch(enc, m.composite, uint32_t(dst.width), uint32_t(dst.height));
    [enc endEncoding];
    return true;
}

bool GlowMetal::submit_render_premultiplied(void* command_queue, const HostFrame& src, const HostFrame& dst,
                                            int work_width, int work_height, const GlowParams& p, bool clamp_premult) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)command_queue;
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        if (!cb) return false;
        if (!encode_render_premultiplied((__bridge void*)cb, src, dst, work_width, work_height, p, clamp_premult)) {
            return false;
        }
        [cb commit];
        return cb.error == nil;
    }
}

bool GlowMetal::render_premultiplied(const Image& rgba, const GlowParams& p, Image& out, Timing* timing) {
    @autoreleasepool {
        const auto t0 = std::chrono::steady_clock::now();
        const int W = rgba.width, H = rgba.height;
        out = Image(W, H, 4);
        const size_t px = size_t(W) * H;
        id<MTLBuffer> in = [impl_->device newBufferWithLength:px * 16 options:MTLResourceStorageModeShared];
        id<MTLBuffer> res = [impl_->device newBufferWithLength:px * 16 options:MTLResourceStorageModeShared];
        float* f = static_cast<float*>(in.contents);
        for (size_t i = 0; i < px; ++i) {  // RGBA -> BGRA
            f[i * 4 + 0] = rgba.data[i * 4 + 2];
            f[i * 4 + 1] = rgba.data[i * 4 + 1];
            f[i * 4 + 2] = rgba.data[i * 4 + 0];
            f[i * 4 + 3] = rgba.data[i * 4 + 3];
        }
        HostFrame frame{(__bridge void*)in, W, H, W, 0, 0};
        HostFrame target{(__bridge void*)res, W, H, W, 0, 0};
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        if (!encode_render_premultiplied((__bridge void*)cb, frame, target, W, H, p, false)) return false;
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return false;
        const float* r = static_cast<const float*>(res.contents);
        for (size_t i = 0; i < px; ++i) {
            out.data[i * 4 + 0] = r[i * 4 + 2];
            out.data[i * 4 + 1] = r[i * 4 + 1];
            out.data[i * 4 + 2] = r[i * 4 + 0];
            out.data[i * 4 + 3] = r[i * 4 + 3];
        }
        if (timing) {
            timing->gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            timing->total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
        return true;
    }
}

}  // namespace metal
}  // namespace glow_profond
