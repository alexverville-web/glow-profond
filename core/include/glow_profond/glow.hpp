// Glow Profond: C++ port of reference/glow.py.
//
// The NumPy reference is the source of truth for the look; this core must
// match it (see tests/parity.py). Images are interleaved float RGBA, rows
// top to bottom, encoded in the comp's working space (sRGB-like gamma
// unless gamma_correct is false).
#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <vector>

namespace glow_profond {

using Rgb = std::array<float, 3>;

// Behind: the glow only shows where the layer is transparent, so text and
// logos stay crisp with the halo behind them.
enum class View { Final, Glow, Source, Behind };

// How the glow is laid over the source: add, or screen (src + glow * (1 - src)
// on channels at or below 1, so whites blow out much less).
enum class Blend { Add, Screen };

struct GlowParams {
    float radius = 400.0f;           // largest blur sigma, in pixels
    float exposure = 0.0f;           // stops applied to the glow
    float threshold = 0.0f;          // in linear light
    float threshold_smooth = 0.5f;   // soft knee, as a fraction of threshold (0..1)
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
    Blend blend = Blend::Add;
    float dirt_amount = 0.0f;        // lens dirt lit by the wide part of the glow
    float depth_spread = 0.0f;       // 0..1: far sources (depth 1) push energy to wide levels
    bool occlusion = false;          // nearer pixels mask the glow of farther sources
    float occlusion_softness = 0.05f;  // depth range of the occlusion smoothstep
    float temporal_smooth = 0.0f;    // 0..1, blends neighbouring frames into the glow source
    float saturation = 1.0f;         // glow saturation around its Rec. 709 luma: 0 grey, 1 as is, 2 double
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

// Pyramid levels for a given radius (matches the reference level_plan):
// blur sigma of each level in source pixels, and its weight.
struct LevelPlan {
    std::vector<float> sigmas;
    std::vector<float> weights;
};
LevelPlan level_plan(const GlowParams& p);
int level_count(const GlowParams& p);

// Glow only, linear light, same size as src (3 channels, premultiplied).
// depth, when given, is read from its first channel (0 = near, 1 = far) and
// drives p.depth_spread and p.occlusion. dirt, when given, is a linear RGB
// texture lit by the untinted wide half of the pyramid, scaled by
// p.dirt_amount. Both are resized bilinearly if their size differs.
Image glow_layer(const Image& src_lin_rgb, const GlowParams& p, const Image* depth = nullptr,
                 const Image* dirt = nullptr);

// Full effect. src has 3 or 4 channels; returns 3 channels, like the
// reference render(). dirt is in the same encoding as src.
Image render(const Image& src, const GlowParams& p, const Image* depth = nullptr, const Image* dirt = nullptr);

// Host-facing variant for After Effects: src is RGBA premultiplied, and so
// is the result. The glow widens the alpha coverage so it shows past the
// layer's edges. For opaque input its RGB equals render().
//
// Optional host inputs, all the same size as src and in the same encoding:
// - depth: depth map, red channel, 0 = near, 1 = far (not linearized; RGBA
//   input is unpremultiplied first);
// - dirt: lens dirt texture (RGBA, straight or premultiplied alike);
// - neighbors: the previous/next frames (RGBA premultiplied), averaged into
//   the glow source by p.temporal_smooth to calm flickering highlights. The
//   source itself is still composited from the current frame only.
struct GlowInputs {
    const Image* depth = nullptr;
    const Image* dirt = nullptr;
    std::vector<const Image*> neighbors;
};
Image render_premultiplied(const Image& src, const GlowParams& p, const GlowInputs& inputs = {});

// Runs fn(row) for every row in [0, rows) on all cores (the core's own
// scheduler, shared with hosts so pixel conversions run in parallel too).
void parallel_for_rows(int rows, const std::function<void(int)>& fn);

// Building blocks, exposed for tests.
Image soft_threshold(const Image& rgb, float t, float smooth);
Image downsample2(const Image& img);
Image upsample_to(const Image& img, int width, int height);
Image upsample_tent2(const Image& img, int width, int height);
void blur_axis(Image& img, int channel, float sigma, bool vertical);
float srgb_to_linear(float x);
float linear_to_srgb(float x);
float aces_fitted(float x);

}  // namespace glow_profond
