// Glow Profond: C++ port of reference/glow.py.
//
// The NumPy reference is the source of truth for the look; this core must
// match it (see tests/parity.py). Images are interleaved float RGBA, rows
// top to bottom, encoded in the comp's working space (sRGB-like gamma
// unless gamma_correct is false).
#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace glow_profond {

using Rgb = std::array<float, 3>;

enum class View { Final, Glow, Source };

struct GlowParams {
    float radius = 400.0f;           // largest blur sigma, in pixels
    float exposure = 0.0f;           // stops applied to the glow
    float threshold = 0.0f;          // in linear light
    float threshold_smooth = 0.25f;  // soft knee width, linear light
    float spread = 2.0f;             // falloff exponent p: glow ~ 1/r^p
    float aspect = 1.0f;             // >1 stretches horizontally, <1 vertically
    Rgb rgb_radius = {1.0f, 1.0f, 1.0f};  // per-channel radius multipliers
    std::vector<Rgb> tint = {{1.0f, 1.0f, 1.0f}};  // core -> halo colours
    bool unmult = false;             // derive alpha from max(R,G,B)
    bool gamma_correct = true;       // linearize before blurring, re-encode after
    bool tonemap = false;            // ACES-fitted curve on the composite
    float base_sigma = 1.5f;         // sigma of the tightest level, in pixels
    int downsample = 1;              // extra global downsample (power of two)
    View view = View::Final;
};

// Planar-free, interleaved image with a fixed channel count.
struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<float> data;

    Image() = default;
    Image(int w, int h, int c) : width(w), height(h), channels(c), data(size_t(w) * h * c, 0.0f) {}

    float* px(int x, int y) { return data.data() + (size_t(y) * width + x) * channels; }
    const float* px(int x, int y) const { return data.data() + (size_t(y) * width + x) * channels; }
};

// Number of pyramid levels used for a given radius (matches the reference).
int level_count(const GlowParams& p);

// Glow only, linear light, same size as src (3 channels, premultiplied).
Image glow_layer(const Image& src_lin_rgb, const GlowParams& p);

// Full effect. src has 3 or 4 channels; returns 3 channels, like the
// reference render().
Image render(const Image& src, const GlowParams& p);

// Host-facing variant for After Effects: src is RGBA premultiplied, and so
// is the result. The glow widens the alpha coverage so it shows past the
// layer's edges. For opaque input its RGB equals render().
Image render_premultiplied(const Image& src, const GlowParams& p);

// Building blocks, exposed for tests.
Image soft_threshold(const Image& rgb, float t, float knee);
Image downsample2(const Image& img);
Image upsample_to(const Image& img, int width, int height);
void blur_axis(Image& img, int channel, float sigma, bool vertical);
float srgb_to_linear(float x);
float linear_to_srgb(float x);
float aces_fitted(float x);

}  // namespace glow_profond
