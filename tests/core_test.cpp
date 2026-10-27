// Property tests for the Glow Profond core (no framework, exits non-zero on failure).
#include <algorithm>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <vector>

#include "glow_profond/glow.hpp"

using namespace glow_profond;

static int failures = 0;

#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                \
        }                                                              \
    } while (0)

static double channel_sum(const Image& img, int c) {
    double s = 0.0;
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x) s += img.px(x, y)[c];
    return s;
}

static void test_blur_conserves_energy() {
    Image img(64, 64, 3);
    img.px(32, 32)[0] = 1.0f;
    blur_axis(img, 0, 3.0f, false);
    blur_axis(img, 0, 3.0f, true);
    CHECK(std::fabs(channel_sum(img, 0) - 1.0) < 1e-5);
    // Symmetric around the impulse.
    CHECK(std::fabs(img.px(29, 32)[0] - img.px(35, 32)[0]) < 1e-7f);
    CHECK(std::fabs(img.px(32, 28)[0] - img.px(32, 36)[0]) < 1e-7f);
}

static void test_downsample_conserves_mean() {
    Image img(8, 6, 3);
    for (size_t i = 0; i < img.data.size(); ++i) img.data[i] = float(i % 7) / 7.0f;
    const Image d = downsample2(img);
    CHECK(d.width == 4 && d.height == 3);
    CHECK(std::fabs(channel_sum(d, 1) * 4.0 - channel_sum(img, 1)) < 1e-4);
    const Image odd = downsample2(Image(5, 3, 3));
    CHECK(odd.width == 3 && odd.height == 2);  // odd sizes round up
}

static void test_tent_upsample_preserves_constant() {
    Image img(7, 5, 3);
    for (float& v : img.data) v = 0.7f;
    const Image up = upsample_tent2(img, 15, 11);  // odd target sizes
    for (float v : up.data) CHECK(std::fabs(v - 0.7f) < 1e-6f);
}

static void test_threshold() {
    Image img(3, 1, 3);
    img.px(0, 0)[0] = 0.1f;   // well below t - knee
    img.px(1, 0)[0] = 2.0f;   // well above t + knee
    img.px(2, 0)[0] = 0.5f;   // inside the knee
    const Image t = soft_threshold(img, 0.5f, 0.2f);  // knee = 0.1
    CHECK(t.px(0, 0)[0] == 0.0f);
    CHECK(std::fabs(t.px(1, 0)[0] - 1.5f) < 1e-6f);
    CHECK(t.px(2, 0)[0] > 0.0f && t.px(2, 0)[0] < 0.5f);

    // Threshold 0 passes everything through, dark pixels included.
    Image dark(2, 1, 3);
    dark.px(0, 0)[1] = 0.001f;
    dark.px(1, 0)[2] = 0.3f;
    const Image same = soft_threshold(dark, 0.0f, 0.5f);
    for (size_t i = 0; i < dark.data.size(); ++i) CHECK(std::fabs(same.data[i] - dark.data[i]) < 1e-7f);
}

static void test_level_count() {
    GlowParams p;
    p.base_sigma = 1.5f;
    p.radius = 1.0f;
    CHECK(level_count(p) == 1);
    p.radius = 3.0f;
    CHECK(level_count(p) == 2);
    p.radius = 400.0f;
    CHECK(level_count(p) == 10);  // ceil(log2(266.7) + 1): the 10th level fades in
    p.radius = 1.5f * 32.0f;  // exactly 6 octaves: all weights 1, last sigma = radius
    const LevelPlan plan = level_plan(p);
    CHECK(plan.sigmas.size() == 6 && std::fabs(plan.sigmas[5] - p.radius) < 1e-3f);
    for (float w : plan.weights) CHECK(std::fabs(w - 1.0f) < 1e-5f);
}

static void test_tonemap_and_srgb() {
    CHECK(aces_fitted(0.0f) < 0.01f);
    CHECK(aces_fitted(100.0f) == 1.0f);
    for (float x = 0.0f; x < 4.0f; x += 0.01f) CHECK(aces_fitted(x + 0.01f) >= aces_fitted(x));
    for (float x = 0.0f; x <= 1.0f; x += 0.05f) CHECK(std::fabs(linear_to_srgb(srgb_to_linear(x)) - x) < 1e-5f);
}

static void test_exposure_off_view_source_is_identity() {
    Image src(16, 16, 3);
    for (size_t i = 0; i < src.data.size(); ++i) src.data[i] = float(i % 11) / 11.0f;
    GlowParams p;
    p.view = View::Source;
    const Image out = render(src, p);
    for (size_t i = 0; i < src.data.size(); ++i) CHECK(std::fabs(out.data[i] - src.data[i]) < 1e-5f);
}

static void test_glow_energy_is_exposure() {
    // Total glow energy is the thresholded input energy times the summed
    // level weights times 2^exposure (away from the borders).
    Image src(256, 256, 3);
    for (int c = 0; c < 3; ++c) src.px(128, 128)[c] = 1.0f;
    GlowParams p;
    p.radius = 20.0f;
    p.threshold_smooth = 1e-5f;
    p.exposure = 1.0f;
    const Image g = glow_layer(src, p);
    double wsum = 0.0;
    for (float w : level_plan(p).weights) wsum += w;
    CHECK(std::fabs(channel_sum(g, 1) - 2.0 * wsum) < 0.01 * wsum);
}

static void test_buffer_size_does_not_move_the_glow() {
    // A host hands us buffers of varying size around the same layer (the
    // padding follows the radius). The glow must not drift or turn lopsided.
    GlowParams p;
    p.radius = 100.0f;
    auto glow_in = [&](int size) {
        Image src(size, size, 3);
        for (int y = 40; y < 90; ++y)
            for (int x = 40; x < 90; ++x)
                for (int c = 0; c < 3; ++c) src.px(x, y)[c] = 1.0f;
        return glow_layer(src, p);
    };
    const Image a = glow_in(257), b = glow_in(300);
    // Only the clamped far edges differ (within a coarse texel of them).
    float worst = 0.0f;
    for (int y = 0; y < 128; ++y)
        for (int x = 0; x < 128; ++x) worst = std::max(worst, std::fabs(a.px(x, y)[0] - b.px(x, y)[0]));
    CHECK(worst < 0.01f * a.px(30, 65)[0]);
    // And it is centred: same glow left and right of the square.
    CHECK(std::fabs(a.px(30, 65)[0] - a.px(99, 65)[0]) < 0.03f * a.px(30, 65)[0]);
}

static void test_radius_grows_the_halo_smoothly() {
    // Sliding the radius up never dims the far halo and never jumps.
    Image src(256, 256, 3);
    for (int c = 0; c < 3; ++c) src.px(128, 128)[c] = 1000.0f;
    GlowParams p;
    p.threshold_smooth = 1e-5f;
    std::vector<float> far;
    for (float r = 20.0f; r <= 70.0f; r += 1.0f) {
        p.radius = r;
        far.push_back(glow_layer(src, p).px(128 + 60, 128)[1]);
    }
    for (size_t i = 1; i < far.size(); ++i) {
        CHECK(far[i] >= far[i - 1] - 1e-6f);
        CHECK(far[i] - far[i - 1] < 0.1f * far.back());  // no jump
    }
}

static void test_premultiplied_matches_render_when_opaque() {
    Image rgb(96, 64, 3), rgba(96, 64, 4);
    for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 96; ++x) {
            const float v = (x > 40 && x < 56 && y > 24 && y < 40) ? 1.0f : 0.05f;
            const float col[3] = {v, v * 0.6f, v * 0.2f};
            for (int c = 0; c < 3; ++c) rgb.px(x, y)[c] = rgba.px(x, y)[c] = col[c];
            rgba.px(x, y)[3] = 1.0f;
        }
    }
    GlowParams p;
    p.radius = 30.0f;
    p.tonemap = true;
    const Image a = render(rgb, p);
    const Image b = render_premultiplied(rgba, p);
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 96; ++x)
            for (int c = 0; c < 3; ++c) CHECK(std::fabs(a.px(x, y)[c] - b.px(x, y)[c]) < 1e-5f);
}

static void test_premultiplied_glow_extends_alpha() {
    Image src(64, 64, 4);
    for (int c = 0; c < 4; ++c) src.px(32, 32)[c] = 1.0f;  // one opaque white pixel
    GlowParams p;
    p.radius = 10.0f;
    p.exposure = 4.0f;
    const Image out = render_premultiplied(src, p);
    CHECK(out.px(32, 32)[3] == 1.0f);
    CHECK(out.px(36, 32)[3] > 0.0f);   // glow spills onto transparent pixels
    CHECK(out.px(0, 0)[3] < 1e-3f);
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x)
            if (out.px(x, y)[3] < 1.0f)  // HDR values above 1 are allowed where opaque
                for (int c = 0; c < 3; ++c) CHECK(out.px(x, y)[c] <= out.px(x, y)[3] + 1e-5f);
}

static Image dot_scene(float v) {
    Image img(64, 64, 4);
    for (int c = 0; c < 3; ++c) img.px(32, 32)[c] = v;
    img.px(32, 32)[3] = 1.0f;
    return img;
}

static void test_screen_composite() {
    Image src(64, 64, 4);
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x) {
            const float v = (x > 24 && x < 40) ? 1.0f : 0.5f;
            for (int c = 0; c < 3; ++c) src.px(x, y)[c] = v;
            src.px(x, y)[3] = 1.0f;
        }
    GlowParams p;
    p.radius = 20.0f;
    p.blend = Blend::Screen;
    p.gamma_correct = false;
    p.view = View::Glow;
    const Image g = render_premultiplied(src, p);
    p.view = View::Final;
    const Image out = render_premultiplied(src, p);
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x)
            for (int c = 0; c < 3; ++c) {
                const float s = src.px(x, y)[c], gl = g.px(x, y)[c];
                CHECK(std::fabs(out.px(x, y)[c] - (s + gl * (1.0f - s))) < 1e-5f);
                if (gl <= 1.0f) CHECK(out.px(x, y)[c] <= 1.0f + 1e-5f);  // whites stay white
            }
    p.blend = Blend::Add;
    p.exposure = 3.0f;
    const Image add = render_premultiplied(src, p);
    CHECK(add.px(32, 32)[0] > 1.0f);  // add does blow past 1
}

static void test_depth_spread_widens_far_sources() {
    Image src(128, 128, 3);
    for (int c = 0; c < 3; ++c) src.px(64, 64)[c] = 1.0f;
    GlowParams p;
    p.radius = 24.0f;  // widest sigma well inside the canvas, so no energy leaks
    Image near(128, 128, 1), far(128, 128, 1);
    for (float& v : far.data) v = 1.0f;
    p.depth_spread = 1.0f;
    const Image gn = glow_layer(src, p, &near), gf = glow_layer(src, p, &far);
    p.depth_spread = 0.0f;
    const Image g0 = glow_layer(src, p, &far);
    for (size_t i = 0; i < g0.data.size(); ++i) CHECK(std::fabs(gn.data[i] - g0.data[i]) < 1e-6f);  // near: no change
    CHECK(gf.px(64, 64)[0] < g0.px(64, 64)[0]);  // far: less core, more halo
    CHECK(gf.px(64 + 30, 64)[0] > g0.px(64 + 30, 64)[0]);
    CHECK(std::fabs(channel_sum(gf, 0) - channel_sum(g0, 0)) < 0.02 * channel_sum(g0, 0));  // same energy
}

static void test_occlusion_masks_far_glow_on_near_pixels() {
    Image src(128, 64, 3), depth(128, 64, 1);
    for (int c = 0; c < 3; ++c) src.px(32, 32)[c] = 1.0f;  // far light on the left
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 128; ++x) depth.px(x, y)[0] = x < 64 ? 1.0f : 0.0f;  // near wall on the right
    GlowParams p;
    p.radius = 60.0f;
    const Image open = glow_layer(src, p, &depth);
    p.occlusion = true;
    const Image occl = glow_layer(src, p, &depth);
    CHECK(occl.px(80, 32)[0] < 1e-3f * open.px(80, 32)[0] + 1e-9f);  // near wall: masked
    CHECK(std::fabs(occl.px(40, 32)[0] - open.px(40, 32)[0]) < 1e-6f);  // same depth: untouched
}

static void test_dirt_only_lights_where_glow_is() {
    GlowParams p;
    p.radius = 40.0f;
    p.view = View::Glow;
    const Image src = dot_scene(1.0f);
    Image dirt(64, 64, 4);
    for (float& v : dirt.data) v = 1.0f;
    GlowInputs in;
    in.dirt = &dirt;
    const Image without = render_premultiplied(src, p, in);  // dirt_amount 0: no effect
    p.dirt_amount = 1.0f;
    const Image with = render_premultiplied(src, p, in);
    CHECK(with.px(40, 32)[0] > without.px(40, 32)[0]);
    const Image none = render_premultiplied(Image(64, 64, 4), p, in);  // no glow, no dirt
    for (float v : none.data) CHECK(v == 0.0f);
}

static void test_temporal_smooth_blends_neighbours() {
    GlowParams p;
    p.radius = 10.0f;
    p.view = View::Glow;
    p.gamma_correct = false;
    const Image on = dot_scene(1.0f), off = dot_scene(0.0f);
    GlowInputs in;
    in.neighbors = {&off, &off};
    const Image raw = render_premultiplied(on, p, in);  // temporal_smooth 0: ignored
    p.temporal_smooth = 1.0f;
    const Image smooth = render_premultiplied(on, p, in);
    // Half the energy comes from the current frame, half from the dark neighbours.
    CHECK(std::fabs(smooth.px(34, 32)[1] - 0.5f * raw.px(34, 32)[1]) < 1e-4f);
}

static void test_saturation() {
    // A pure red source: saturation 0 makes the glow grey (equal channels),
    // saturation 1 changes nothing, and the result is never negative.
    Image src(64, 64, 3);
    src.px(32, 32)[0] = 1.0f;
    GlowParams p;
    p.radius = 8.0f;
    const Image base = glow_layer(src, p);
    p.saturation = 0.0f;
    const Image grey = glow_layer(src, p);
    const float* g = grey.px(34, 32);
    CHECK(std::fabs(g[0] - g[1]) < 1e-7f && std::fabs(g[1] - g[2]) < 1e-7f && g[0] > 0.0f);
    p.saturation = 3.0f;
    const Image vivid = glow_layer(src, p);
    for (float v : vivid.data) CHECK(v >= 0.0f);
    CHECK(vivid.px(34, 32)[0] > base.px(34, 32)[0]);
}

static void test_behind_keeps_opaque_pixels() {
    // Opaque pixels are untouched by the glow; transparent ones receive it.
    Image src(64, 64, 4);
    for (int y = 28; y < 36; ++y)
        for (int x = 28; x < 36; ++x)
            for (int c = 0; c < 4; ++c) src.px(x, y)[c] = 1.0f;
    GlowParams p;
    p.radius = 10.0f;
    p.exposure = 2.0f;
    p.view = View::Behind;
    const Image out = render_premultiplied(src, p);
    for (int c = 0; c < 4; ++c) CHECK(std::fabs(out.px(31, 31)[c] - 1.0f) < 1e-5f);
    CHECK(out.px(40, 32)[3] > 0.0f && out.px(40, 32)[0] > 0.0f);
}

static void test_transparent_layer_over_black_matches_opaque() {
    // A white shape on a transparent layer, composited over black (premult
    // colour as is), must look like render() of the same shape on opaque
    // black. The old encoding kept the glow linear in alpha: the wide halo
    // vanished in After Effects.
    const int W = 160, H = 96;
    Image rgba(W, H, 4), rgb(W, H, 3);
    for (int y = 30; y < 60; ++y)
        for (int x = 40; x < 80; ++x) {
            for (int c = 0; c < 4; ++c) rgba.px(x, y)[c] = 1.0f;
            for (int c = 0; c < 3; ++c) rgb.px(x, y)[c] = 1.0f;
        }
    for (int tonemap = 0; tonemap < 2; ++tonemap) {
        GlowParams p;
        p.radius = 60.0f;
        p.tonemap = tonemap != 0;
        const Image a = render(rgb, p);
        const Image b = render_premultiplied(rgba, p);
        float worst = 0.0f;
        int invalid = 0;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                for (int c = 0; c < 3; ++c) worst = std::max(worst, std::fabs(a.px(x, y)[c] - b.px(x, y)[c]));
                // Valid premultiplied pixel wherever it isn't opaque (opaque HDR may exceed 1).
                if (b.px(x, y)[3] < 1.0f) invalid += b.px(x, y)[0] > b.px(x, y)[3] + 1e-6f;
            }
        CHECK(worst < 1e-5f);
        CHECK(invalid == 0);
    }
}

int main() {
    test_transparent_layer_over_black_matches_opaque();
    test_screen_composite();
    test_depth_spread_widens_far_sources();
    test_occlusion_masks_far_glow_on_near_pixels();
    test_saturation();
    test_behind_keeps_opaque_pixels();
    test_dirt_only_lights_where_glow_is();
    test_temporal_smooth_blends_neighbours();
    test_premultiplied_matches_render_when_opaque();
    test_premultiplied_glow_extends_alpha();
    test_blur_conserves_energy();
    test_downsample_conserves_mean();
    test_tent_upsample_preserves_constant();
    test_threshold();
    test_level_count();
    test_radius_grows_the_halo_smoothly();
    test_buffer_size_does_not_move_the_glow();
    test_tonemap_and_srgb();
    test_exposure_off_view_source_is_identity();
    test_glow_energy_is_exposure();
    if (failures == 0) std::printf("all core tests passed\n");
    return failures == 0 ? 0 : 1;
}
