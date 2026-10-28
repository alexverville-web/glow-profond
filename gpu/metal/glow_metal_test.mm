// Compares the Metal glow with the CPU core on the same input (the plain
// path: no depth, no lens dirt), and times both. Exit code 1 if any case differs by more than the tolerance.
//   glow_metal_test [--bench]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "GlowMetal.h"

using glow_profond::GlowParams;
using glow_profond::Image;

namespace {

constexpr float kTolerance = 1e-3f;

// Linear-light test frame: a soft gradient, hot spots and thin lines, so
// thresholds, odd sizes and every pyramid level get exercised.
Image scene(int w, int h) {
    Image img(w, h, 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float* o = img.px(x, y);
            const float g = 0.05f + 0.1f * float(x) / float(w);
            o[0] = g;
            o[1] = g * 0.8f;
            o[2] = g * 0.6f;
            if (((x / 97) + (y / 61)) % 9 == 0) {
                o[0] = 4.0f;
                o[1] = 2.0f;
                o[2] = 0.5f;
            }
            if (y % 211 == 7) o[2] = 3.0f;
        }
    }
    return img;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    std::string error;
    auto gpu = glow_profond::metal::GlowMetal::create(&error);
    if (!gpu) {
        std::printf("Metal unavailable: %s\n", error.c_str());
        return 1;
    }

    struct Case {
        const char* name;
        GlowParams p;
    };
    std::vector<Case> cases;
    auto add = [&](const char* name, auto&& tweak) {
        GlowParams p;
        p.radius = 200.0f;
        tweak(p);
        cases.push_back({name, p});
    };
    add("default", [](GlowParams&) {});
    add("radius_400", [](GlowParams& p) { p.radius = 400.0f; });
    add("tiny_radius", [](GlowParams& p) { p.radius = 1.0f; });
    add("spread_3_exposure", [](GlowParams& p) { p.spread = 3.0f; p.exposure = 1.5f; });
    add("anamorphic", [](GlowParams& p) { p.aspect = 6.0f; });
    add("chromatic", [](GlowParams& p) { p.rgb_radius = {1.4f, 1.0f, 0.7f}; });
    add("multitint", [](GlowParams& p) { p.tint = {{1, 1, 1}, {1, 0.4f, 0.8f}, {0.3f, 0.3f, 1}}; });
    add("threshold", [](GlowParams& p) { p.threshold = 1.0f; p.threshold_smooth = 0.2f; });
    add("downsample_2", [](GlowParams& p) { p.downsample = 2; });
    add("downsample_8", [](GlowParams& p) { p.downsample = 8; p.radius = 600.0f; });
    add("wide_base_sigma", [](GlowParams& p) { p.base_sigma = 8.0f; p.aspect = 10.0f; p.rgb_radius = {4, 1, 1}; });
    add("saturation", [](GlowParams& p) { p.saturation = 1.8f; });
    add("desaturated", [](GlowParams& p) { p.saturation = 0.0f; });
    add("downsample_sat", [](GlowParams& p) { p.downsample = 4; p.saturation = 0.5f; });
    add("quality_8_r160", [](GlowParams& p) { p.downsample = 8; p.radius = 160.0f; p.spread = 1.85f; });
    add("core_size", [](GlowParams& p) { p.base_sigma = 5.0f; p.base_sigma_ref = 1.5f; });
    // Without a depth or dirt image these must be no-ops, like on the CPU.
    add("depth_dirt_unset", [](GlowParams& p) { p.depth_spread = 1.0f; p.occlusion = true; p.dirt_amount = 1.0f; });

    int failures = 0;
    const int sizes[][2] = {{641, 359}, {1920, 1080}};
    for (const auto& sz : sizes) {
        const Image src = scene(sz[0], sz[1]);
        for (const Case& c : cases) {
            const Image ref = glow_profond::glow_layer(src, c.p);
            Image out;
            if (!gpu->glow_layer(src, c.p, out)) {
                std::printf("FAIL %-18s %dx%d: Metal run failed\n", c.name, sz[0], sz[1]);
                ++failures;
                continue;
            }
            float max_diff = 0.0f, max_val = 0.0f;
            for (size_t i = 0; i < ref.data.size(); ++i) {
                max_diff = std::max(max_diff, std::fabs(ref.data[i] - out.data[i]));
                max_val = std::max(max_val, std::fabs(ref.data[i]));
            }
            const bool ok = max_diff <= kTolerance;
            failures += !ok;
            std::printf("%s %-18s %4dx%-4d max|diff| = %.2e (max value %.2f)\n", ok ? "ok  " : "FAIL", c.name, sz[0],
                        sz[1], max_diff, max_val);
        }
    }

    // Full render_premultiplied (the After Effects path): a transparent layer
    // with shapes, soft edges and an HDR spot, every view and blend.
    {
        const int W = 803, H = 451;
        Image rgba(W, H, 4);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float* o = rgba.px(x, y);
                const float dx = float(x - 250), dy = float(y - 220);
                const float disc = std::clamp(60.0f - std::sqrt(dx * dx + dy * dy), 0.0f, 1.0f);  // soft edge
                const bool bar = x > 450 && x < 700 && y > 200 && y < 230;
                const float a = std::max(disc, bar ? 1.0f : 0.0f);
                const float r = bar ? 0.2f : 1.0f, g = bar ? 0.7f : 0.6f, b = bar ? 1.0f : 0.2f;
                o[0] = r * a;
                o[1] = g * a;
                o[2] = b * a;
                o[3] = a;
                if (x == 600 && y == 300) { o[0] = o[1] = o[2] = 8.0f; o[3] = 1.0f; }  // HDR spot (32 bpc)
            }
        struct RCase {
            const char* name;
            GlowParams p;
        };
        std::vector<RCase> rc;
        auto addr = [&](const char* name, auto&& tweak) {
            GlowParams p;
            p.radius = 160.0f;
            p.exposure = -1.25f;
            p.spread = 1.85f;
            tweak(p);
            rc.push_back({name, p});
        };
        addr("final", [](GlowParams&) {});
        addr("glow_only", [](GlowParams& p) { p.view = glow_profond::View::Glow; });
        addr("source", [](GlowParams& p) { p.view = glow_profond::View::Source; });
        addr("behind_screen", [](GlowParams& p) { p.view = glow_profond::View::Behind; p.blend = glow_profond::Blend::Screen; });
        addr("tonemap_unmult", [](GlowParams& p) { p.tonemap = true; p.unmult = true; });
        addr("linear_sat", [](GlowParams& p) { p.gamma_correct = false; p.saturation = 1.6f; });
        addr("hable_mix_opacity", [](GlowParams& p) {
            p.tonemap = true;
            p.tonemap_curve = glow_profond::Tonemap::Hable;
            p.tonemap_mix = 0.6f;
            p.source_opacity = 0.35f;
        });
        addr("reinhard_quality4", [](GlowParams& p) {
            p.tonemap = true;
            p.tonemap_curve = glow_profond::Tonemap::Reinhard;
            p.downsample = 4;
        });
        for (const RCase& c : rc) {
            const Image ref = glow_profond::render_premultiplied(rgba, c.p);
            Image out;
            if (!gpu->render_premultiplied(rgba, c.p, out)) {
                std::printf("FAIL render %-14s: Metal run failed\n", c.name);
                ++failures;
                continue;
            }
            float max_diff = 0.0f;
            for (size_t i = 0; i < ref.data.size(); ++i) max_diff = std::max(max_diff, std::fabs(ref.data[i] - out.data[i]));
            const bool ok = max_diff <= kTolerance;
            failures += !ok;
            std::printf("%s render %-14s %dx%d max|diff| = %.2e\n", ok ? "ok  " : "FAIL", c.name, W, H, max_diff);
        }
    }

    if (bench) {
        const int bsizes[][2] = {{1920, 1080}, {3840, 2160}};
        for (const auto& sz : bsizes) {
            const Image src = scene(sz[0], sz[1]);
            GlowParams p;  // radius 400, the plug-in default
            double cpu = 1e9, gpu_ms = 1e9, total = 1e9;
            for (int i = 0; i < 5; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                const Image ref = glow_profond::glow_layer(src, p);
                cpu = std::min(cpu, ms_since(t0));
                Image out;
                glow_profond::metal::Timing t;
                gpu->glow_layer(src, p, out, &t);
                gpu_ms = std::min(gpu_ms, t.gpu_ms);
                total = std::min(total, t.total_ms);
            }
            std::printf("bench %dx%d radius 400: CPU glow_layer %.1f ms | Metal GPU %.1f ms (%.1f ms with upload/readback)\n",
                        sz[0], sz[1], cpu, gpu_ms, total);
            // The whole effect as After Effects runs it (defaults, transparent layer).
            Image rgba(sz[0], sz[1], 4);
            for (size_t i = 0; i < src.data.size() / 3; ++i) {
                const float a = std::min(src.data[i * 3], 1.0f);
                for (int c = 0; c < 3; ++c) rgba.data[i * 4 + c] = std::min(src.data[i * 3 + c], 1.0f) * a;
                rgba.data[i * 4 + 3] = a;
            }
            GlowParams q;
            q.radius = 160.0f;
            q.exposure = -1.25f;
            q.spread = 1.85f;
            double cpu_r = 1e9, gpu_r = 1e9;
            for (int i = 0; i < 5; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                const Image ref = glow_profond::render_premultiplied(rgba, q);
                cpu_r = std::min(cpu_r, ms_since(t0));
                Image out;
                glow_profond::metal::Timing t;
                gpu->render_premultiplied(rgba, q, out, &t);
                gpu_r = std::min(gpu_r, t.gpu_ms);
            }
            std::printf("bench %dx%d full effect (defaults): CPU %.1f ms | Metal GPU %.1f ms\n", sz[0], sz[1], cpu_r, gpu_r);
        }
    }
    return failures ? 1 : 0;
}
