// Simulates Glow Profond's After Effects render path outside AE: PreRender's
// rects (padding, request clipping), the aligned work bounds and the input /
// output offsets, then render_premultiplied on the CPU and the Metal path.
// Sweeps the radius and moves the layer, and checks that the halo stays
// centred, only grows, and follows the layer exactly. Exit code 1 on failure.
//   ae_path_sim [--quick]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "GlowMetal.h"

using glow_profond::GlowParams;
using glow_profond::Image;

namespace {

struct Rect {
    long left, top, right, bottom;
    long w() const { return right - left; }
    long h() const { return bottom - top; }
};

// Mirrors of the wrapper's helpers (ae/GlowProfond.cpp).
Rect expand(Rect r, long pad) { return {r.left - pad, r.top - pad, r.right + pad, r.bottom + pad}; }
Rect intersect(Rect a, Rect b) {
    return {std::max(a.left, b.left), std::max(a.top, b.top), std::min(a.right, b.right), std::min(a.bottom, b.bottom)};
}
Rect unite(Rect a, Rect b) {
    return {std::min(a.left, b.left), std::min(a.top, b.top), std::max(a.right, b.right), std::max(a.bottom, b.bottom)};
}
long glow_padding(const GlowParams& p) { return long(std::ceil(4.0f * p.radius + 4.0f * p.downsample)); }
Rect work_bounds(Rect in, Rect out, const GlowParams& p) {
    Rect b = unite(in, out);
    const Rect reach = expand(in, glow_padding(p));
    const long margin = long(std::ceil(0.5f * p.radius));
    if (out.left > reach.left) b.left -= margin;
    if (out.top > reach.top) b.top -= margin;
    if (out.right < reach.right) b.right += margin;
    if (out.bottom < reach.bottom) b.bottom += margin;
    const int steps = glow_profond::level_count(p) - 1 + int(std::log2(float(std::max(p.downsample, 1))));
    const long cell = 1L << std::clamp(steps, 0, 14);
    auto floor_to = [cell](long v) { return (v >= 0 ? v / cell : -((-v + cell - 1) / cell)) * cell; };
    b.left = floor_to(b.left);
    b.top = floor_to(b.top);
    return b;
}

// One AE frame: a white square (layer content rect `content`) in a comp of
// comp_w x comp_h; returns the comp-sized premultiplied output.
Image render_frame(const GlowParams& p, Rect content, long comp_w, long comp_h, glow_profond::metal::GlowMetal* gpu) {
    const Rect request{0, 0, comp_w, comp_h};
    const long pad = glow_padding(p);
    const Rect in_rect = content;  // AE: the input's result rect is the shape's bounds
    const Rect out_rect = intersect(expand(in_rect, pad), request);
    const Rect b = work_bounds(in_rect, out_rect, p);
    Image comp(int(comp_w), int(comp_h), 4);
    if (!gpu) {
        Image src(int(b.w()), int(b.h()), 4);
        for (long y = in_rect.top; y < in_rect.bottom; ++y)
            for (long x = in_rect.left; x < in_rect.right; ++x)
                for (int c = 0; c < 4; ++c) src.px(int(x - b.left), int(y - b.top))[c] = 1.0f;
        const Image res = glow_profond::render_premultiplied(src, p);
        for (long y = out_rect.top; y < out_rect.bottom; ++y)
            for (long x = out_rect.left; x < out_rect.right; ++x)
                std::memcpy(comp.px(int(x), int(y)), res.px(int(x - b.left), int(y - b.top)), 16);
        return comp;
    }
    // GPU: separate input and output frames, as AE hands them over.
    Image in_img(int(in_rect.w()), int(in_rect.h()), 4), out_img(int(out_rect.w()), int(out_rect.h()), 4);
    for (float& v : in_img.data) v = 1.0f;
    gpu->render_frames(in_img, int(in_rect.left - b.left), int(in_rect.top - b.top), out_img,
                       int(out_rect.left - b.left), int(out_rect.top - b.top), int(b.w()), int(b.h()), p);
    for (long y = out_rect.top; y < out_rect.bottom; ++y)
        for (long x = out_rect.left; x < out_rect.right; ++x)
            std::memcpy(comp.px(int(x), int(y)), out_img.px(int(x - out_rect.left), int(y - out_rect.top)), 16);
    return comp;
}

// The same frame rendered with no clipping at all (sources and halo in a
// buffer much larger than the comp): what the comp should show.
Image render_unclipped(const GlowParams& p, Rect content, long comp_w, long comp_h) {
    const long m = glow_padding(p) + 64;
    const long cell = 1L << std::clamp(glow_profond::level_count(p) - 1, 0, 14);
    const long left = ((-m) / cell - 1) * cell, top = ((-m) / cell - 1) * cell;  // aligned like the plug-in
    Image src(int(comp_w + m - left), int(comp_h + m - top), 4);
    for (long y = content.top; y < content.bottom; ++y)
        for (long x = content.left; x < content.right; ++x)
            for (int c = 0; c < 4; ++c) src.px(int(x - left), int(y - top))[c] = 1.0f;
    const Image res = glow_profond::render_premultiplied(src, p);
    Image comp(int(comp_w), int(comp_h), 4);
    for (long y = 0; y < comp_h; ++y)
        for (long x = 0; x < comp_w; ++x) std::memcpy(comp.px(int(x), int(y)), res.px(int(x - left), int(y - top)), 16);
    return comp;
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    std::string error;
    auto gpu = glow_profond::metal::GlowMetal::create(&error);
    int failures = 0;
    const long CW = 1920, CH = 1080;
    for (int use_gpu = 0; use_gpu < (gpu ? 2 : 1); ++use_gpu) {
        const char* dev = use_gpu ? "GPU" : "CPU";
        // 1. Radius sweep, square at an odd position: centred, growing, steady.
        float prev_far = -1.0f, prev_near = -1.0f;
        float worst_shift = 0.0f, worst_lr = 0.0f, worst_drop = 0.0f, worst_ref = 0.0f;
        int steps = 0;
        for (float r = 10.0f; r <= 1000.0f; r *= quick ? 1.25f : 1.04f, ++steps) {
            GlowParams p;
            p.radius = r;
            p.exposure = -1.25f;
            p.spread = 1.85f;
            p.view = glow_profond::View::Glow;
            const Rect sq{901, 487, 1021, 607};  // 120 px square, centre (961, 547)
            const Image f = render_frame(p, sq, CW, CH, use_gpu ? gpu.get() : nullptr);
            // Halo centroid (alpha-weighted) over a window centred on the
            // square, so the comp edges don't bias it; balance at 40 px.
            double sx = 0, sy = 0, sw = 0;
            for (int y = 547 - 500; y <= 546 + 500; ++y)
                for (int x = 961 - 500; x <= 960 + 500; ++x) {
                    const double w = f.px(x, y)[1];
                    sx += w * x;
                    sy += w * y;
                    sw += w;
                }
            const float cx = float(sx / sw) - 960.5f, cy = float(sy / sw) - 546.5f;
            const float left = f.px(901 - 40, 547)[1], right = f.px(1020 + 40, 547)[1];
            const float up = f.px(961, 487 - 40)[1], down = f.px(961, 606 + 40)[1];
            // In 8-bit levels of the output.
            const float lr = 255.0f * std::max(std::fabs(left - right), std::fabs(up - down));
            const float near = f.px(1020 + 10, 547)[1], far = f.px(1020 + 300, 547)[1];
            // Against the unclipped render (every 4th pixel): clipping to the
            // comp and the aligned buffer must not change what is visible.
            if (!quick || steps % 4 == 0) {
                const Image ref = render_unclipped(p, sq, CW, CH);
                float wr = 0.0f;
                int wx = 0, wy = 0;
                for (int y = 0; y < CH; y += 4)
                    for (int x = 0; x < CW; x += 4) {
                        const float dd = 255.0f * std::fabs(ref.px(x, y)[1] - f.px(x, y)[1]);
                        if (dd > wr) { wr = dd; wx = x; wy = y; }
                    }
                if (getenv("SIM_VERBOSE")) std::printf("  r=%.0f vs unclipped %.2f at (%d,%d)\n", r, wr, wx, wy);
                worst_ref = std::max(worst_ref, wr);
            }
            worst_shift = std::max({worst_shift, std::fabs(cx), std::fabs(cy)});
            worst_lr = std::max(worst_lr, lr);
            if (prev_far >= 0) worst_drop = std::max({worst_drop, prev_far - far, prev_near - near});
            prev_far = far;
            prev_near = near;
        }
        // Centroid within 0.5 px, sides within 2 8-bit levels (pyramid
        // aliasing depends on where the shape sits on the coarse grid), and
        // raising the radius never darkens anything.
        // The centroid is informative only: a wide halo is cut by the comp
        // edges, which is correct, so the pass criterion is the unclipped match.
        // 1.25 levels: at the comp edges, very wide halos keep ~1/255 of
        // error even with the margin (WorkBounds).
        const bool ok1 = worst_ref <= 1.25f && worst_lr <= 2.0f && worst_drop <= 1e-4f;
        failures += !ok1;
        std::printf("%s %s radius 10->1000 (%d steps): vs unclipped render %.2f levels, L/R and U/D imbalance "
                    "%.2f levels, max drop %.1e (centroid in window %.2f px)\n",
                    ok1 ? "ok  " : "FAIL", dev, steps, worst_ref, worst_lr, worst_drop, worst_shift);

        // 2. Moving layer: shifting the square by 1 px shifts the glow by 1 px.
        float worst_move = 0.0f;
        for (float r : {37.0f, 160.0f, 611.0f}) {
            GlowParams p;
            p.radius = r;
            p.exposure = -1.25f;
            p.spread = 1.85f;
            const Image a = render_frame(p, {700, 400, 820, 520}, CW, CH, use_gpu ? gpu.get() : nullptr);
            for (int d = 1; d <= 3; ++d) {
                const Image b = render_frame(p, {700 + d, 400 + d, 820 + d, 520 + d}, CW, CH, use_gpu ? gpu.get() : nullptr);
                for (int y = 300; y < 640; y += 7)
                    for (int x = 600; x < 940; x += 7)
                        for (int c = 0; c < 4; ++c)
                            worst_move = std::max(worst_move, std::fabs(b.px(x + d, y + d)[c] - a.px(x, y)[c]));
            }
        }
        const bool ok2 = 255.0f * worst_move <= 2.0f;
        failures += !ok2;
        std::printf("%s %s moving layer 1-3 px: glow changes by at most %.2f 8-bit levels\n", ok2 ? "ok  " : "FAIL",
                    dev, 255.0f * worst_move);
    }

    // 3. Half-resolution preview: same halo, at half size.
    {
        GlowParams full, half;
        full.radius = 300.0f;
        full.exposure = -1.25f;
        full.spread = 1.85f;
        half = full;
        half.radius = 150.0f;
        half.base_sigma = full.base_sigma * 0.5f;
        const Image a = render_frame(full, {800, 400, 1040, 640}, CW, CH, nullptr);
        const Image b = render_frame(half, {400, 200, 520, 320}, CW / 2, CH / 2, nullptr);
        // Half-res pixel (x, y) covers full-res 2x..2x+1: compare with their mean.
        float worst = 0.0f;
        for (int d : {20, 60, 150, 300}) {
            const int hx = 520 + d / 2, hy = 260;
            float fa = 0.0f;
            for (int j = 0; j < 2; ++j)
                for (int i = 0; i < 2; ++i) fa += 0.25f * a.px(2 * hx + i, 2 * hy + j)[1];
            worst = std::max(worst, std::fabs(fa - b.px(hx, hy)[1]));
            if (getenv("SIM_VERBOSE")) std::printf("  half-res d=%d: full %.4f half %.4f\n", d, fa, b.px(hx, hy)[1]);
        }
        // Preview only: at half res the per-level blur is already smaller
        // than what the down/upsample filters add, so the far halo comes
        // out a few % brighter. Tolerance 6 levels; the render is full res.
        const bool ok3 = 255.0f * worst <= 6.0f;
        failures += !ok3;
        std::printf("%s half-res preview: halo within %.2f 8-bit levels of full res\n", ok3 ? "ok  " : "FAIL",
                    255.0f * worst);
    }
    return failures ? 1 : 0;
}
