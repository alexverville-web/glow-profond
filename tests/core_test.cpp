// Property tests for the Glow Profond core (no framework, exits non-zero on failure).
#include <cmath>
#include <cstdio>

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
    CHECK(odd.width == 2 && odd.height == 1);
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
    CHECK(level_count(p) == 9);  // floor(log2(266.7)) + 1
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
    // Equal-weight levels are normalized, so total glow energy equals the
    // thresholded input energy times 2^exposure (away from the borders).
    Image src(256, 256, 3);
    for (int c = 0; c < 3; ++c) src.px(128, 128)[c] = 1.0f;
    GlowParams p;
    p.radius = 20.0f;
    p.threshold_smooth = 1e-5f;
    p.exposure = 1.0f;
    const Image g = glow_layer(src, p);
    CHECK(std::fabs(channel_sum(g, 1) - 2.0) < 0.02);
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

static void test_screen_never_exceeds_one() {
    Image src(64, 64, 4);
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x) {
            const float v = (x > 24 && x < 40) ? 1.0f : 0.5f;
            for (int c = 0; c < 3; ++c) src.px(x, y)[c] = v;
            src.px(x, y)[3] = 1.0f;
        }
    GlowParams p;
    p.radius = 20.0f;
    p.exposure = 3.0f;
    p.blend = Blend::Screen;
    p.gamma_correct = false;
    const Image out = render_premultiplied(src, p);
    for (float v : out.data) CHECK(v <= 1.0f + 1e-5f);
    p.blend = Blend::Add;
    const Image add = render_premultiplied(src, p);
    CHECK(add.px(32, 32)[0] > 1.0f);  // add does blow past 1
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

int main() {
    test_screen_never_exceeds_one();
    test_dirt_only_lights_where_glow_is();
    test_temporal_smooth_blends_neighbours();
    test_premultiplied_matches_render_when_opaque();
    test_premultiplied_glow_extends_alpha();
    test_blur_conserves_energy();
    test_downsample_conserves_mean();
    test_tent_upsample_preserves_constant();
    test_threshold();
    test_level_count();
    test_tonemap_and_srgb();
    test_exposure_off_view_source_is_identity();
    test_glow_energy_is_exposure();
    if (failures == 0) std::printf("all core tests passed\n");
    return failures == 0 ? 0 : 1;
}
