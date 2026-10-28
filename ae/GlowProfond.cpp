// Glow Profond: After Effects SmartFX wrapper around the glow core.
//
// The look lives in core/ (a port of reference/glow.py); this file only maps
// AE parameters and pixel buffers onto glow_profond::render_premultiplied.
// 8, 16 and 32 bpc are supported; buffers are converted to float RGBA.
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <exception>
#include <new>

#include "AEConfig.h"
#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_EffectSuites.h"
#include "AE_Macros.h"
#include "AEFX_SuiteHelper.h"
#include "Param_Utils.h"

#include "GlowProfond_Version.h"
#include "glow_profond/glow.hpp"

#ifdef AE_OS_MAC
#include "AE_EffectGPUSuites.h"
#include "AE_EffectPixelFormat.h"
#include "GlowMetal.h"
#endif

static_assert(GP_OUT_FLAGS == (PF_OutFlag_DEEP_COLOR_AWARE | PF_OutFlag_I_EXPAND_BUFFER | PF_OutFlag_WIDE_TIME_INPUT |
                               PF_OutFlag_SEND_UPDATE_PARAMS_UI),
              "PiPL out flags out of sync with GlobalSetup");
static_assert(GP_OUT_FLAGS_2 == (PF_OutFlag2_FLOAT_COLOR_AWARE | PF_OutFlag2_SUPPORTS_SMART_RENDER |
                                 PF_OutFlag2_SUPPORTS_THREADED_RENDERING |
                                 PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG |
                                 PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT | PF_OutFlag2_SUPPORTS_GPU_RENDER_F32),
              "PiPL out flags 2 out of sync with GlobalSetup");
static_assert(GP_VERSION_PACKED == PF_VERSION(GP_MAJOR_VERSION, GP_MINOR_VERSION, GP_BUG_VERSION,
                                              GP_STAGE_VERSION, GP_BUILD_VERSION),
              "PiPL version out of sync with my_version");

namespace {

using glow_profond::GlowParams;
using glow_profond::Image;

// Parameter order in the Effect Controls panel: the essentials at the top,
// every extra behind an "Enable" checkbox in its own (collapsed) group.
enum {
    GP_INPUT = 0,
    GP_PRESET,
    GP_RADIUS,
    GP_EXPOSURE,
    GP_THRESHOLD,
    GP_SPREAD_AMOUNT,
    GP_SATURATION,
    GP_BLEND,
    GP_VIEW,
    GP_TINT_GROUP,
    GP_TINT_MODE,
    GP_TINT_CORE,
    GP_TINT_MID,
    GP_TINT_HALO,
    GP_TINT_GROUP_END,
    GP_CHROMA_GROUP,
    GP_CHROMA_ENABLE,
    GP_RADIUS_R,
    GP_RADIUS_G,
    GP_RADIUS_B,
    GP_CHROMA_GROUP_END,
    GP_ANAMORPHIC_GROUP,
    GP_ANAMORPHIC_ENABLE,
    GP_ASPECT,
    GP_ANAMORPHIC_GROUP_END,
    GP_DIRT_GROUP,
    GP_DIRT_ENABLE,
    GP_DIRT_LAYER,
    GP_DIRT_AMOUNT,
    GP_DIRT_GROUP_END,
    GP_DEPTH_GROUP,
    GP_DEPTH_ENABLE,
    GP_DEPTH_LAYER,
    GP_DEPTH_SPREAD,
    GP_OCCLUSION,
    GP_DEPTH_INVERT,
    GP_DEPTH_GROUP_END,
    GP_FLICKER_GROUP,
    GP_FLICKER_ENABLE,
    GP_TEMPORAL,
    GP_FLICKER_GROUP_END,
    GP_RENDER_GROUP,
    GP_THRESHOLD_SMOOTH,
    GP_UNMULT,
    GP_GAMMA,
    GP_TONEMAP,
    GP_BASE_SIGMA,
    GP_DOWNSAMPLE,
    GP_RENDER_GROUP_END,
    GP_NUM_PARAMS
};

// Disk IDs are saved in projects: never renumber, only append. IDs of
// parameters that left the panel (the raw falloff exponent, the old Input
// and Shape groups) stay reserved.
enum {
    ID_RADIUS = 1,
    ID_EXPOSURE,
    ID_SPREAD,  // v0.1-0.3 raw exponent, replaced by ID_SPREAD_AMOUNT
    ID_INPUT_GROUP,
    ID_THRESHOLD,
    ID_THRESHOLD_SMOOTH,
    ID_UNMULT,
    ID_INPUT_GROUP_END,
    ID_SHAPE_GROUP,
    ID_ASPECT,
    ID_RADIUS_R,
    ID_RADIUS_G,
    ID_RADIUS_B,
    ID_SHAPE_GROUP_END,
    ID_TINT_GROUP,
    ID_TINT_MODE,
    ID_TINT_CORE,
    ID_TINT_MID,
    ID_TINT_HALO,
    ID_TINT_GROUP_END,
    ID_RENDER_GROUP,
    ID_GAMMA,
    ID_TONEMAP,
    ID_BASE_SIGMA,
    ID_DOWNSAMPLE,
    ID_RENDER_GROUP_END,
    ID_VIEW,
    // v0.2
    ID_PRESET,
    ID_SATURATION,
    ID_BLEND,
    ID_TEMPORAL,
    ID_DIRT_GROUP,
    ID_DIRT_LAYER,
    ID_DIRT_AMOUNT,
    ID_DIRT_GROUP_END,
    // v0.3
    ID_DEPTH_GROUP,
    ID_DEPTH_LAYER,
    ID_DEPTH_SPREAD,
    ID_OCCLUSION,
    ID_DEPTH_INVERT,
    ID_DEPTH_GROUP_END,
    // v0.4
    ID_SPREAD_AMOUNT,
    ID_CHROMA_GROUP,
    ID_CHROMA_ENABLE,
    ID_CHROMA_GROUP_END,
    ID_ANAMORPHIC_GROUP,
    ID_ANAMORPHIC_ENABLE,
    ID_ANAMORPHIC_GROUP_END,
    ID_DIRT_ENABLE,
    ID_DEPTH_ENABLE,
    ID_FLICKER_GROUP,
    ID_FLICKER_ENABLE,
    ID_FLICKER_GROUP_END
};

enum { TINT_NONE = 1, TINT_TWO, TINT_THREE };
enum { VIEW_FINAL = 1, VIEW_GLOW, VIEW_SOURCE, VIEW_BEHIND };
enum { BLEND_ADD = 1, BLEND_SCREEN };
enum { PRESET_CUSTOM = 1, PRESET_NEON, PRESET_SOFT_SUN, PRESET_CINE, PRESET_ANAMORPHIC, PRESET_TEXT, PRESET_SUBTLE,
       PRESET_COUNT = PRESET_SUBTLE };

// Spread (0-100 %, more = wider) to the core's falloff exponent: 100 % is
// 1/r (widest), 50 % 1/r^2, 0 % 1/r^3 (tightest).
float SpreadToExponent(float percent) { return 3.0f - 2.0f * std::clamp(percent, 0.0f, 100.0f) / 100.0f; }

// Checkout ids for the extra layer requests (distinct from parameter indices).
enum { CHECKOUT_PREV = 1000, CHECKOUT_NEXT, CHECKOUT_DIRT, CHECKOUT_DEPTH };

// What PreRender decided, handed to SmartRender.
struct RenderPlan {
    PF_LRect input_rect;   // layer-space rect covered by the input world
    PF_LRect output_rect;  // layer-space rect covered by the output world
    PF_LRect prev_rect;    // neighbouring frames (anti-flicker), empty if unused
    PF_LRect next_rect;
    bool dirt = false;     // a lens dirt layer was checked out
    PF_LRect dirt_rect;    // in the dirt layer's own space
    bool depth = false;    // a depth layer was checked out
    PF_LRect depth_rect;   // in the depth layer's own space
    A_long layer_width;    // effect layer size at render resolution, for the dirt mapping
    A_long layer_height;
};

void DeleteRenderPlan(void* plan) { delete static_cast<RenderPlan*>(plan); }

bool IsEmpty(const PF_LRect& r) { return r.left >= r.right || r.top >= r.bottom; }

PF_LRect Expand(PF_LRect r, A_long pad) {
    if (IsEmpty(r)) return r;
    r.left -= pad;
    r.top -= pad;
    r.right += pad;
    r.bottom += pad;
    return r;
}

PF_LRect Intersect(const PF_LRect& a, const PF_LRect& b) {
    PF_LRect r;
    r.left = std::max(a.left, b.left);
    r.top = std::max(a.top, b.top);
    r.right = std::min(a.right, b.right);
    r.bottom = std::min(a.bottom, b.bottom);
    if (IsEmpty(r)) r.left = r.top = r.right = r.bottom = 0;
    return r;
}

PF_LRect Union(const PF_LRect& a, const PF_LRect& b) {
    if (IsEmpty(a)) return b;
    if (IsEmpty(b)) return a;
    PF_LRect r;
    r.left = std::min(a.left, b.left);
    r.top = std::min(a.top, b.top);
    r.right = std::max(a.right, b.right);
    r.bottom = std::max(a.bottom, b.bottom);
    return r;
}

// The look was calibrated on Deep Glow's default render with the core at
// exposure -1.25; the Exposure slider shows that look as 0.
constexpr float kExposureCalibration = -1.25f;

A_long GlowPadding(const GlowParams& p);

// Work area for a render: input and output rects, plus
// - where AE's request cut the output short of the glow's reach (comp
//   edges), a margin of 0.5 x radius: the coarse levels need texels past
//   the visible edge, else the halo is off by up to 9/255 along the frame
//   at large radii (measured against an unclipped render);
// - its top-left moved down to a multiple of the coarsest pyramid cell
//   (2^(levels - 1) pixels, times the global downsample) in layer
//   coordinates, so the levels sit on the same grid for every radius and
//   every requested rect: the halo can't shift or wobble when the buffer
//   grows with the radius.
PF_LRect WorkBounds(const PF_LRect& input, const PF_LRect& output, const GlowParams& p) {
    PF_LRect b = Union(input, output);
    if (IsEmpty(b) || p.view == glow_profond::View::Source) return b;
    const PF_LRect reach = Expand(input, GlowPadding(p));
    const A_long margin = A_long(std::ceil(0.5f * p.radius));
    if (output.left > reach.left) b.left -= margin;
    if (output.top > reach.top) b.top -= margin;
    if (output.right < reach.right) b.right += margin;
    if (output.bottom < reach.bottom) b.bottom += margin;
    const int steps = glow_profond::level_count(p) - 1 + int(std::log2(float(std::max(p.downsample, 1))));
    const A_long cell = A_long(1) << std::clamp(steps, 0, 14);
    auto floor_to = [cell](A_long v) { return (v >= 0 ? v / cell : -((-v + cell - 1) / cell)) * cell; };
    b.left = floor_to(b.left);
    b.top = floor_to(b.top);
    return b;
}

float ToFloat(const PF_ParamDef& def) { return static_cast<float>(def.u.fs_d.value); }

// Reads every parameter at the current time. Pixel sizes are scaled to the
// render resolution, so a half-resolution preview matches the full render.
// invert_depth, when given, receives the "Invert Depth" checkbox,
// which is a host-side option (not a GlowParams field).
PF_Err ReadParams(PF_InData* in_data, PF_OutData* out_data, GlowParams& p, bool* invert_depth = nullptr) {
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    static const int kValueParams[] = {
        GP_RADIUS,       GP_EXPOSURE,     GP_THRESHOLD,     GP_SPREAD_AMOUNT, GP_SATURATION,   GP_BLEND,
        GP_VIEW,         GP_TINT_MODE,    GP_TINT_CORE,     GP_TINT_MID,      GP_TINT_HALO,    GP_CHROMA_ENABLE,
        GP_RADIUS_R,     GP_RADIUS_G,     GP_RADIUS_B,      GP_ANAMORPHIC_ENABLE, GP_ASPECT,   GP_DIRT_ENABLE,
        GP_DIRT_AMOUNT,  GP_DEPTH_ENABLE, GP_DEPTH_SPREAD,  GP_OCCLUSION,     GP_DEPTH_INVERT, GP_FLICKER_ENABLE,
        GP_TEMPORAL,     GP_THRESHOLD_SMOOTH, GP_UNMULT,    GP_GAMMA,         GP_TONEMAP,      GP_BASE_SIGMA,
        GP_DOWNSAMPLE};
    PF_ParamDef defs[GP_NUM_PARAMS];
    bool checked_out[GP_NUM_PARAMS] = {};
    for (int index : kValueParams) {
        AEFX_CLR_STRUCT(defs[index]);
        ERR(PF_CHECKOUT_PARAM(in_data, index, in_data->current_time, in_data->time_step,
                              in_data->time_scale, &defs[index]));
        if (!err) checked_out[index] = true;
    }

    if (!err) {
        const float scale = static_cast<float>(in_data->downsample_x.num) /
                            static_cast<float>(std::max<A_long>(in_data->downsample_x.den, 1));
        auto on = [&](int index) { return defs[index].u.bd.value != 0; };
        p.radius = ToFloat(defs[GP_RADIUS]) * scale;
        p.base_sigma = ToFloat(defs[GP_BASE_SIGMA]) * scale;
        // Core Size is relative to its default (1.5 px): a bigger core folds
        // the finer octaves' light into the core instead of dropping it.
        p.base_sigma_ref = 1.5f * scale;
        p.exposure = ToFloat(defs[GP_EXPOSURE]) + kExposureCalibration;
        p.spread = SpreadToExponent(ToFloat(defs[GP_SPREAD_AMOUNT]));
        p.threshold = ToFloat(defs[GP_THRESHOLD]);
        p.threshold_smooth = ToFloat(defs[GP_THRESHOLD_SMOOTH]);
        p.unmult = on(GP_UNMULT);
        p.gamma_correct = on(GP_GAMMA);
        p.tonemap = on(GP_TONEMAP);
        p.downsample = 1 << std::clamp<A_long>(defs[GP_DOWNSAMPLE].u.pd.value - 1, 0, 3);
        p.saturation = ToFloat(defs[GP_SATURATION]) / 100.0f;
        p.blend = defs[GP_BLEND].u.pd.value == BLEND_SCREEN ? glow_profond::Blend::Screen : glow_profond::Blend::Add;
        // Extras only count while their group is enabled.
        p.aspect = on(GP_ANAMORPHIC_ENABLE) ? ToFloat(defs[GP_ASPECT]) : 1.0f;
        p.rgb_radius = on(GP_CHROMA_ENABLE)
                           ? glow_profond::Rgb{ToFloat(defs[GP_RADIUS_R]), ToFloat(defs[GP_RADIUS_G]),
                                               ToFloat(defs[GP_RADIUS_B])}
                           : glow_profond::Rgb{1.0f, 1.0f, 1.0f};
        p.temporal_smooth = on(GP_FLICKER_ENABLE) ? ToFloat(defs[GP_TEMPORAL]) / 100.0f : 0.0f;
        p.dirt_amount = on(GP_DIRT_ENABLE) ? ToFloat(defs[GP_DIRT_AMOUNT]) / 100.0f : 0.0f;
        // Also need a depth layer (see SmartRender).
        p.depth_spread = on(GP_DEPTH_ENABLE) ? ToFloat(defs[GP_DEPTH_SPREAD]) : 0.0f;
        p.occlusion = on(GP_DEPTH_ENABLE) && on(GP_OCCLUSION);
        if (invert_depth) *invert_depth = on(GP_DEPTH_INVERT);

        switch (defs[GP_VIEW].u.pd.value) {
            case VIEW_GLOW: p.view = glow_profond::View::Glow; break;
            case VIEW_SOURCE: p.view = glow_profond::View::Source; break;
            case VIEW_BEHIND: p.view = glow_profond::View::Behind; break;
            default: p.view = glow_profond::View::Final; break;
        }

        AEFX_SuiteScoper<PF_ColorParamSuite1> colors(in_data, kPFColorParamSuite, kPFColorParamSuiteVersion1,
                                                      out_data);
        auto color = [&](int index) {
            PF_PixelFloat c = {};
            colors->PF_GetFloatingPointColorFromColorDef(in_data->effect_ref, &defs[index], &c);
            return glow_profond::Rgb{c.red, c.green, c.blue};
        };
        const A_long mode = defs[GP_TINT_MODE].u.pd.value;
        if (mode == TINT_TWO) {
            p.tint = {color(GP_TINT_CORE), color(GP_TINT_HALO)};
        } else if (mode == TINT_THREE) {
            p.tint = {color(GP_TINT_CORE), color(GP_TINT_MID), color(GP_TINT_HALO)};
        } else {
            p.tint = {{1.0f, 1.0f, 1.0f}};
        }
    }

    for (int index : kValueParams) {
        if (checked_out[index]) ERR2(PF_CHECKIN_PARAM(in_data, &defs[index]));
    }
    return err;
}

// How far the glow can reach beyond its source, in pixels: four sigmas of
// the widest level, stretched by aspect and per-channel radius. Three were
// not enough: once sRGB-encoded, the tail at 3 sigma is still ~2/255, and
// the glow ended on a faint visible box.
A_long GlowPadding(const GlowParams& p) {
    if (p.view == glow_profond::View::Source) return 0;
    const float max_mult = std::max({p.rgb_radius[0], p.rgb_radius[1], p.rgb_radius[2], 1.0f});
    const float stretch = std::max(std::sqrt(p.aspect), 1.0f / std::sqrt(p.aspect));
    const float reach = 4.0f * p.radius * max_mult * stretch + 4.0f * p.downsample;
    return static_cast<A_long>(std::ceil(std::min(reach, 30000.0f)));
}

// Copies world into img at (dx, dy); whatever falls outside img is dropped
// (neighbouring frames can cover a different rect than the current one).
// Integer worlds get colour <= alpha: AE can hand over edge pixels with full
// colour and almost no alpha (seen on a shape layer's stroke join in 16 bpc),
// and dividing by that alpha turned them into blinding lights. A straight
// 8/16-bit colour can't exceed 1, so the clamp only removes the bad pixels;
// float worlds keep HDR values as they are.
template <class Pixel>
void ReadPixels(const PF_EffectWorld* world, float max_value, Image& img, int dx, int dy) {
    const bool clamp_premult = max_value > 1.0f;
    const int x_lo = std::max(0, -dx), x_hi = std::min(int(world->width), img.width - dx);
    glow_profond::parallel_for_rows(int(world->height), [&](int y) {
        if (y + dy < 0 || y + dy >= img.height) return;
        const Pixel* row = reinterpret_cast<const Pixel*>(reinterpret_cast<const char*>(world->data) + y * world->rowbytes);
        for (int x = x_lo; x < x_hi; ++x) {
            float* o = img.px(x + dx, y + dy);
            o[3] = float(row[x].alpha) / max_value;
            const float cap = clamp_premult ? o[3] : 1e30f;
            o[0] = std::min(float(row[x].red) / max_value, cap);
            o[1] = std::min(float(row[x].green) / max_value, cap);
            o[2] = std::min(float(row[x].blue) / max_value, cap);
        }
    });
}

// ReadPixels for whichever depth AE handed us.
PF_Err ReadWorld(PF_PixelFormat format, const PF_EffectWorld* world, Image& img, int dx, int dy) {
    switch (format) {
        case PF_PixelFormat_ARGB128: ReadPixels<PF_PixelFloat>(world, 1.0f, img, dx, dy); return PF_Err_NONE;
        case PF_PixelFormat_ARGB64: ReadPixels<PF_Pixel16>(world, float(PF_MAX_CHAN16), img, dx, dy); return PF_Err_NONE;
        case PF_PixelFormat_ARGB32: ReadPixels<PF_Pixel8>(world, float(PF_MAX_CHAN8), img, dx, dy); return PF_Err_NONE;
        default: return PF_Err_BAD_CALLBACK_PARAM;
    }
}

// Stretches a texture layer (lens dirt, depth) over the effect layer, sampled for each
// pixel of the work buffer whose top-left is at layer (left, top).
Image ResampleLayer(const Image& tex, A_long layer_w, A_long layer_h, A_long left, A_long top, int width, int height) {
    Image out(width, height, 4);
    if (tex.width == 0 || tex.height == 0 || layer_w <= 0 || layer_h <= 0) return out;
    const float sx = float(tex.width) / float(layer_w), sy = float(tex.height) / float(layer_h);
    glow_profond::parallel_for_rows(height, [&](int y) {
        const float v = std::clamp((float(top + y) + 0.5f) * sy - 0.5f, 0.0f, float(tex.height - 1));
        const int y0 = int(v), y1 = std::min(y0 + 1, tex.height - 1);
        const float fy = v - float(y0);
        for (int x = 0; x < width; ++x) {
            const float u = std::clamp((float(left + x) + 0.5f) * sx - 0.5f, 0.0f, float(tex.width - 1));
            const int x0 = int(u), x1 = std::min(x0 + 1, tex.width - 1);
            const float fx = u - float(x0);
            float* o = out.px(x, y);
            for (int c = 0; c < 4; ++c) {
                const float t = tex.px(x0, y0)[c] + (tex.px(x1, y0)[c] - tex.px(x0, y0)[c]) * fx;
                const float b = tex.px(x0, y1)[c] + (tex.px(x1, y1)[c] - tex.px(x0, y1)[c]) * fx;
                o[c] = t + (b - t) * fy;
            }
        }
    });
    return out;
}

// 8x8 Bayer matrix: an ordered dither breaks the banding that wide, smooth
// halos show once quantized to 8 (or 16) bits, without frame-to-frame noise.
const unsigned char kBayer8[8][8] = {
    {0, 32, 8, 40, 2, 34, 10, 42},   {48, 16, 56, 24, 50, 18, 58, 26}, {12, 44, 4, 36, 14, 46, 6, 38},
    {60, 28, 52, 20, 62, 30, 54, 22}, {3, 35, 11, 43, 1, 33, 9, 41},   {51, 19, 59, 27, 49, 17, 57, 25},
    {15, 47, 7, 39, 13, 45, 5, 37},  {63, 31, 55, 23, 61, 29, 53, 21}};

// Integer formats (quantize = true) are dithered, clamped and kept valid
// premultiplied (colour <= alpha). (ox, oy) is the output's layer-space
// origin, so the pattern stays put when the output rect moves.
template <class Pixel, class Channel>
void WritePixels(const Image& img, int dx, int dy, float max_value, bool quantize, A_long ox, A_long oy,
                 PF_EffectWorld* world) {
    glow_profond::parallel_for_rows(int(world->height), [&](int y) {
        Pixel* row = reinterpret_cast<Pixel*>(reinterpret_cast<char*>(world->data) + y * world->rowbytes);
        for (A_long x = 0; x < world->width; ++x) {
            const float* s = img.px(int(x) + dx, y + dy);
            if (!quantize) {
                row[x].red = static_cast<Channel>(s[0]);
                row[x].green = static_cast<Channel>(s[1]);
                row[x].blue = static_cast<Channel>(s[2]);
                row[x].alpha = static_cast<Channel>(s[3]);
                continue;
            }
            const float bias = (float(kBayer8[(y + oy) & 7][(x + ox) & 7]) + 0.5f) / 64.0f;
            auto q = [&](float v) { return std::floor(std::clamp(v, 0.0f, 1.0f) * max_value + bias); };
            const float a = q(s[3]);
            row[x].alpha = static_cast<Channel>(a);
            row[x].red = static_cast<Channel>(std::min(q(s[0]), a));
            row[x].green = static_cast<Channel>(std::min(q(s[1]), a));
            row[x].blue = static_cast<Channel>(std::min(q(s[2]), a));
        }
    });
}

PF_Err About(PF_InData* in_data, PF_OutData* out_data) {
    PF_SPRINTF(out_data->return_msg, "%s v%d.%d.%d\rGimmick Studio\rPhysically based glow (1/r^p falloff).", GP_NAME,
               GP_MAJOR_VERSION, GP_MINOR_VERSION, GP_BUG_VERSION);
    return PF_Err_NONE;
}

PF_Err GlobalSetup(PF_OutData* out_data) {
    out_data->my_version = GP_VERSION_PACKED;
    out_data->out_flags = GP_OUT_FLAGS;
    out_data->out_flags2 = GP_OUT_FLAGS_2;
    return PF_Err_NONE;
}

PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data) {
    PF_Err err = PF_Err_NONE;
    PF_ParamDef def;
    const PF_ParamFlags supervised = PF_ParamFlag_SUPERVISE;
    auto group = [&](const char* name, A_long id) {
        AEFX_CLR_STRUCT(def);
        def.flags = PF_ParamFlag_START_COLLAPSED;
        PF_ADD_TOPIC(name, id);
        return err;
    };
    auto end_group = [&](A_long id) {
        AEFX_CLR_STRUCT(def);
        PF_END_TOPIC(id);
        return err;
    };

    // Defaults: a white shape glows like with Deep Glow's defaults (fitted
    // on renders of both), with a continuous radius.
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUPX("Preset", PRESET_COUNT, PRESET_CUSTOM,
                  "Custom|Neon|Soft Sun|Cinematic|Anamorphic|Glowing Text|Subtle Bloom",
                  supervised | PF_ParamFlag_CANNOT_TIME_VARY, ID_PRESET);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Radius", 0, 4000, 0, 1000, 160, PF_Precision_TENTHS, 0, 0, ID_RADIUS);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Exposure", -10, 10, -5, 5, 0, PF_Precision_HUNDREDTHS, 0, 0, ID_EXPOSURE);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Threshold", 0, 100, 0, 1, 0, PF_Precision_HUNDREDTHS, 0, 0, ID_THRESHOLD);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Spread", 0, 100, 0, 100, 57.5, PF_Precision_TENTHS, PF_ValueDisplayFlag_PERCENT, 0,
                         ID_SPREAD_AMOUNT);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Saturation", 0, 300, 0, 200, 100, PF_Precision_INTEGER, PF_ValueDisplayFlag_PERCENT, 0,
                         ID_SATURATION);
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Blend Mode", 2, BLEND_ADD, "Add|Screen", ID_BLEND);
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("View", 4, VIEW_FINAL, "Final|Glow Only|Source|Glow Behind", ID_VIEW);

    ERR(group("Color Tint", ID_TINT_GROUP));
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUPX("Tint", 3, TINT_NONE, "Off|2 Colors|3 Colors", supervised, ID_TINT_MODE);
    AEFX_CLR_STRUCT(def);
    PF_ADD_COLOR("Core Color", 255, 255, 255, ID_TINT_CORE);
    AEFX_CLR_STRUCT(def);
    PF_ADD_COLOR("Middle Color", 255, 102, 204, ID_TINT_MID);
    AEFX_CLR_STRUCT(def);
    PF_ADD_COLOR("Halo Color", 77, 77, 255, ID_TINT_HALO);
    ERR(end_group(ID_TINT_GROUP_END));

    ERR(group("Chromatic Aberration", ID_CHROMA_GROUP));
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Enable Chromatic Aberration", FALSE, supervised, ID_CHROMA_ENABLE);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Red Radius", 0, 4, 0.5, 2, 1.15, PF_Precision_HUNDREDTHS, 0, 0, ID_RADIUS_R);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Green Radius", 0, 4, 0.5, 2, 1, PF_Precision_HUNDREDTHS, 0, 0, ID_RADIUS_G);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Blue Radius", 0, 4, 0.5, 2, 0.85, PF_Precision_HUNDREDTHS, 0, 0, ID_RADIUS_B);
    ERR(end_group(ID_CHROMA_GROUP_END));

    ERR(group("Anamorphic", ID_ANAMORPHIC_GROUP));
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Enable Anamorphic", FALSE, supervised, ID_ANAMORPHIC_ENABLE);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Aspect Ratio", 0.1, 10, 0.25, 8, 4, PF_Precision_HUNDREDTHS, 0, 0, ID_ASPECT);
    ERR(end_group(ID_ANAMORPHIC_GROUP_END));

    ERR(group("Lens Dirt", ID_DIRT_GROUP));
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Enable Lens Dirt", FALSE, supervised, ID_DIRT_ENABLE);
    AEFX_CLR_STRUCT(def);
    PF_ADD_LAYER("Dirt Texture", PF_LayerDefault_NONE, ID_DIRT_LAYER);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Dirt Amount", 0, 1000, 0, 200, 100, PF_Precision_INTEGER, PF_ValueDisplayFlag_PERCENT, 0,
                         ID_DIRT_AMOUNT);
    ERR(end_group(ID_DIRT_GROUP_END));

    ERR(group("Depth", ID_DEPTH_GROUP));
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Enable Depth", FALSE, supervised, ID_DEPTH_ENABLE);
    AEFX_CLR_STRUCT(def);
    PF_ADD_LAYER("Depth Layer", PF_LayerDefault_NONE, ID_DEPTH_LAYER);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Depth Spread", 0, 1, 0, 1, 0.5, PF_Precision_HUNDREDTHS, 0, 0, ID_DEPTH_SPREAD);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Occlusion", FALSE, 0, ID_OCCLUSION);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Invert Depth (White = Near)", FALSE, 0, ID_DEPTH_INVERT);
    ERR(end_group(ID_DEPTH_GROUP_END));

    ERR(group("Anti-Flicker", ID_FLICKER_GROUP));
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Enable Anti-Flicker", FALSE, supervised, ID_FLICKER_ENABLE);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Smoothing", 0, 100, 0, 100, 50, PF_Precision_INTEGER, PF_ValueDisplayFlag_PERCENT, 0,
                         ID_TEMPORAL);
    ERR(end_group(ID_FLICKER_GROUP_END));

    ERR(group("Advanced", ID_RENDER_GROUP));
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Threshold Smoothness", 0, 1, 0, 1, 0.5, PF_Precision_HUNDREDTHS, 0, 0,
                         ID_THRESHOLD_SMOOTH);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Unmultiply (Black = Transparent)", FALSE, 0, ID_UNMULT);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Linear Light (Gamma Correct)", TRUE, 0, ID_GAMMA);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("ACES Tonemap", FALSE, 0, ID_TONEMAP);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Core Size", 0.5, 8, 0.5, 4, 1.5, PF_Precision_HUNDREDTHS, 0, 0, ID_BASE_SIGMA);
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Quality", 4, 1, "Full|High (1/2)|Medium (1/4)|Draft (1/8)", ID_DOWNSAMPLE);
    ERR(end_group(ID_RENDER_GROUP_END));

    out_data->num_params = GP_NUM_PARAMS;
    return err;
}

// What a preset sets. View, quality, lens dirt, depth, anti-flicker, unmult
// and gamma are left as the user had them.
struct Preset {
    float radius, exposure, spread, saturation;  // spread and saturation in %
    A_long blend;
    float threshold, threshold_smooth;
    bool anamorphic;
    float aspect;
    bool chroma;
    float rgb_radius[3];
    A_long tint_mode;
    A_u_char core[3], mid[3], halo[3];
    bool tonemap;
};

// Indexed by PRESET_* - 2 (Custom sets nothing). Weights are per octave, so
// big radii add light: wide presets use lower exposures. Exposures are UI
// values (kExposureCalibration is added at render). Thresholds stay
// below 0.7 so they still catch highlights in ordinary (SDR) footage.
// Checked on build-mac/test/presets_v04.png.
const Preset kPresets[] = {
    // Neon: tight, saturated, colour kept.
    {120, 0.5f, 50, 140, BLEND_ADD, 0.3f, 0.5f, false, 4, false, {1.15f, 1, 0.85f}, TINT_NONE,
     {255, 255, 255}, {255, 102, 204}, {77, 77, 255}, false},
    // Soft Sun: huge warm halo, screened so whites hold.
    {500, -0.5f, 70, 90, BLEND_SCREEN, 0.4f, 0.8f, false, 4, false, {1.15f, 1, 0.85f}, TINT_TWO,
     {255, 250, 235}, {255, 200, 140}, {255, 170, 90}, true},
    // Cinematic: highlights only, a hint of chromatic spread, ACES.
    {300, 0.25f, 55, 100, BLEND_ADD, 0.5f, 0.6f, false, 4, true, {1.15f, 1, 0.85f}, TINT_NONE,
     {255, 255, 255}, {255, 102, 204}, {77, 77, 255}, true},
    // Anamorphic: wide horizontal streak, blue halo.
    {400, 0.5f, 55, 120, BLEND_ADD, 0.4f, 0.5f, true, 6, false, {1.15f, 1, 0.85f}, TINT_TWO,
     {255, 255, 255}, {160, 190, 255}, {80, 140, 255}, true},
    // Glowing Text: whole shape glows, close and bright.
    {100, 0.75f, 50, 120, BLEND_ADD, 0, 0.5f, false, 4, false, {1.15f, 1, 0.85f}, TINT_NONE,
     {255, 255, 255}, {255, 102, 204}, {77, 77, 255}, false},
    // Subtle Bloom: a soft lift of the brightest areas.
    {250, -0.75f, 60, 100, BLEND_SCREEN, 0.6f, 0.5f, false, 4, false, {1.15f, 1, 0.85f}, TINT_NONE,
     {255, 255, 255}, {255, 102, 204}, {77, 77, 255}, false},
};
static_assert(sizeof(kPresets) / sizeof(kPresets[0]) == PRESET_COUNT - 1, "one entry per preset");

PF_Err UserChangedParam(PF_ParamDef* params[], const PF_UserChangedParamExtra* extra) {
    if (extra->param_index != GP_PRESET) return PF_Err_NONE;
    const A_long choice = params[GP_PRESET]->u.pd.value;
    if (choice <= PRESET_CUSTOM || choice > PRESET_COUNT) return PF_Err_NONE;
    const Preset& pr = kPresets[choice - 2];

    auto set_float = [&](int index, float v) {
        params[index]->u.fs_d.value = v;
        params[index]->uu.change_flags |= PF_ChangeFlag_CHANGED_VALUE;
    };
    auto set_popup = [&](int index, A_long v) {
        params[index]->u.pd.value = v;
        params[index]->uu.change_flags |= PF_ChangeFlag_CHANGED_VALUE;
    };
    auto set_check = [&](int index, bool v) {
        params[index]->u.bd.value = v;
        params[index]->uu.change_flags |= PF_ChangeFlag_CHANGED_VALUE;
    };
    auto set_color = [&](int index, const A_u_char (&c)[3]) {
        params[index]->u.cd.value.red = c[0];
        params[index]->u.cd.value.green = c[1];
        params[index]->u.cd.value.blue = c[2];
        params[index]->u.cd.value.alpha = 255;
        params[index]->uu.change_flags |= PF_ChangeFlag_CHANGED_VALUE;
    };
    set_float(GP_RADIUS, pr.radius);
    set_float(GP_EXPOSURE, pr.exposure);
    set_float(GP_SPREAD_AMOUNT, pr.spread);
    set_float(GP_SATURATION, pr.saturation);
    set_popup(GP_BLEND, pr.blend);
    set_float(GP_THRESHOLD, pr.threshold);
    set_float(GP_THRESHOLD_SMOOTH, pr.threshold_smooth);
    set_check(GP_ANAMORPHIC_ENABLE, pr.anamorphic);
    set_float(GP_ASPECT, pr.aspect);
    set_check(GP_CHROMA_ENABLE, pr.chroma);
    set_float(GP_RADIUS_R, pr.rgb_radius[0]);
    set_float(GP_RADIUS_G, pr.rgb_radius[1]);
    set_float(GP_RADIUS_B, pr.rgb_radius[2]);
    set_popup(GP_TINT_MODE, pr.tint_mode);
    set_color(GP_TINT_CORE, pr.core);
    set_color(GP_TINT_MID, pr.mid);
    set_color(GP_TINT_HALO, pr.halo);
    set_check(GP_TONEMAP, pr.tonemap);
    return PF_Err_NONE;
}

// Greys out the controls of every group that is switched off, so the panel
// shows at a glance what is active.
PF_Err UpdateParamsUI(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[]) {
    PF_Err err = PF_Err_NONE;
    AEFX_SuiteScoper<PF_ParamUtilsSuite3> utils(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_data);
    auto enable = [&](int index, bool on) {
        const bool disabled = (params[index]->ui_flags & PF_PUI_DISABLED) != 0;
        if (disabled != on) return;  // already right
        PF_ParamDef copy = *params[index];
        if (on) {
            copy.ui_flags &= ~PF_PUI_DISABLED;
        } else {
            copy.ui_flags |= PF_PUI_DISABLED;
        }
        ERR(utils->PF_UpdateParamUI(in_data->effect_ref, index, &copy));
    };
    const A_long tint = params[GP_TINT_MODE]->u.pd.value;
    enable(GP_TINT_CORE, tint != TINT_NONE);
    enable(GP_TINT_MID, tint == TINT_THREE);
    enable(GP_TINT_HALO, tint != TINT_NONE);
    const bool chroma = params[GP_CHROMA_ENABLE]->u.bd.value != 0;
    for (int i : {GP_RADIUS_R, GP_RADIUS_G, GP_RADIUS_B}) enable(i, chroma);
    enable(GP_ASPECT, params[GP_ANAMORPHIC_ENABLE]->u.bd.value != 0);
    const bool dirt = params[GP_DIRT_ENABLE]->u.bd.value != 0;
    for (int i : {GP_DIRT_LAYER, GP_DIRT_AMOUNT}) enable(i, dirt);
    const bool depth = params[GP_DEPTH_ENABLE]->u.bd.value != 0;
    for (int i : {GP_DEPTH_LAYER, GP_DEPTH_SPREAD, GP_OCCLUSION, GP_DEPTH_INVERT}) enable(i, depth);
    enable(GP_TEMPORAL, params[GP_FLICKER_ENABLE]->u.bd.value != 0);
    return err;
}

PF_Err PreRender(PF_InData* in_data, PF_OutData* out_data, PF_PreRenderExtra* extra) {
    PF_Err err = PF_Err_NONE;
    GlowParams p;
    ERR(ReadParams(in_data, out_data, p));
    if (err) return err;

    const A_long pad = GlowPadding(p);

    // Any input pixel within `pad` of the requested area can light it up.
    PF_RenderRequest req = extra->input->output_request;
    req.rect = Expand(req.rect, pad);
    PF_CheckoutResult in_result;
    AEFX_CLR_STRUCT(in_result);
    ERR(extra->cb->checkout_layer(in_data->effect_ref, GP_INPUT, GP_INPUT, &req, in_data->current_time,
                                  in_data->time_step, in_data->time_scale, &in_result));
    if (err) return err;

    // The glow spreads past the layer, but we only render what was asked.
    extra->output->result_rect = Intersect(Expand(in_result.result_rect, pad), extra->input->output_request.rect);
    extra->output->max_result_rect = Expand(in_result.max_result_rect, pad);

    RenderPlan* plan = new (std::nothrow) RenderPlan;
    if (!plan) return PF_Err_OUT_OF_MEMORY;
    plan->input_rect = in_result.result_rect;
    plan->output_rect = extra->output->result_rect;
    plan->prev_rect = plan->next_rect = plan->dirt_rect = plan->depth_rect = PF_LRect{0, 0, 0, 0};
    plan->dirt = plan->depth = false;
    plan->layer_width = A_long(std::lround(double(in_data->width) * in_data->downsample_x.num /
                                           std::max<A_long>(in_data->downsample_x.den, 1)));
    plan->layer_height = A_long(std::lround(double(in_data->height) * in_data->downsample_y.num /
                                            std::max<A_long>(in_data->downsample_y.den, 1)));
    const bool glows = p.view != glow_profond::View::Source;

    // Anti-flicker: the frames before and after feed the glow source. With
    // PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT, AE tracks these dependencies.
    if (!err && glows && p.temporal_smooth > 0.0f) {
        const A_long times[2] = {in_data->current_time - in_data->time_step,
                                 in_data->current_time + in_data->time_step};
        PF_LRect* rects[2] = {&plan->prev_rect, &plan->next_rect};
        for (int i = 0; i < 2 && !err; ++i) {
            PF_CheckoutResult r;
            AEFX_CLR_STRUCT(r);
            err = extra->cb->checkout_layer(in_data->effect_ref, GP_INPUT, CHECKOUT_PREV + i, &req, times[i],
                                            in_data->time_step, in_data->time_scale, &r);
            if (!err) *rects[i] = r.result_rect;
        }
    }
    // Lens dirt and depth: whole layers, stretched over the effect layer.
    auto checkout_whole = [&](int param, int id, bool& used, PF_LRect& rect) {
        PF_RenderRequest wreq = req;
        wreq.rect = PF_LRect{0, 0, 30000, 30000};  // clipped by AE to the layer's bounds
        PF_CheckoutResult r;
        AEFX_CLR_STRUCT(r);
        err = extra->cb->checkout_layer(in_data->effect_ref, param, id, &wreq, in_data->current_time,
                                        in_data->time_step, in_data->time_scale, &r);
        if (!err && !IsEmpty(r.result_rect)) {
            used = true;
            rect = r.result_rect;
        }
    };
    if (!err && glows && p.dirt_amount > 0.0f) checkout_whole(GP_DIRT_LAYER, CHECKOUT_DIRT, plan->dirt, plan->dirt_rect);
    if (!err && glows && (p.depth_spread > 0.0f || p.occlusion)) {
        checkout_whole(GP_DEPTH_LAYER, CHECKOUT_DEPTH, plan->depth, plan->depth_rect);
    }
    if (err) {
        delete plan;
        return err;
    }
    // The GPU path covers everything but the extra inputs (depth, lens
    // dirt, neighbouring frames); with those, AE renders on the CPU.
    const bool extra_inputs = plan->dirt || plan->depth || !IsEmpty(plan->prev_rect) || !IsEmpty(plan->next_rect);
    if (!extra_inputs) extra->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
    extra->output->pre_render_data = plan;
    extra->output->delete_pre_render_data_func = DeleteRenderPlan;
    return err;
}

PF_Err SmartRender(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra) {
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    const RenderPlan* plan = static_cast<const RenderPlan*>(extra->input->pre_render_data);
    if (!plan) return PF_Err_BAD_CALLBACK_PARAM;

    GlowParams p;
    bool invert_depth = false;
    ERR(ReadParams(in_data, out_data, p, &invert_depth));

    PF_EffectWorld* input = nullptr;
    PF_EffectWorld* output = nullptr;
    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, GP_INPUT, &input));
    ERR(extra->cb->checkout_output(in_data->effect_ref, &output));

    if (!err && input && output) {
        AEFX_SuiteScoper<PF_WorldSuite2> worlds(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
        PF_PixelFormat format = PF_PixelFormat_INVALID;
        ERR(worlds->PF_GetPixelFormat(output, &format));

        // Work on the union of input and output so the glow sees its sources.
        const PF_LRect bounds = WorkBounds(plan->input_rect, plan->output_rect, p);
        Image src(int(bounds.right - bounds.left), int(bounds.bottom - bounds.top), 4);
        const int in_dx = int(plan->input_rect.left - bounds.left);
        const int in_dy = int(plan->input_rect.top - bounds.top);
        const int out_dx = int(plan->output_rect.left - bounds.left);
        const int out_dy = int(plan->output_rect.top - bounds.top);
        // Guard against rects and worlds disagreeing (should not happen).
        const bool fits = in_dx >= 0 && in_dy >= 0 && out_dx >= 0 && out_dy >= 0 &&
                          in_dx + input->width <= src.width && in_dy + input->height <= src.height &&
                          out_dx + output->width <= src.width && out_dy + output->height <= src.height;

        if (!err && !fits) err = PF_Err_BAD_CALLBACK_PARAM;
        if (!err && src.width > 0 && src.height > 0) {
            switch (format) {
                case PF_PixelFormat_ARGB128:
                    ReadPixels<PF_PixelFloat>(input, 1.0f, src, in_dx, in_dy);
                    break;
                case PF_PixelFormat_ARGB64:
                    ReadPixels<PF_Pixel16>(input, float(PF_MAX_CHAN16), src, in_dx, in_dy);
                    break;
                case PF_PixelFormat_ARGB32:
                    ReadPixels<PF_Pixel8>(input, float(PF_MAX_CHAN8), src, in_dx, in_dy);
                    break;
                default:
                    err = PF_Err_BAD_CALLBACK_PARAM;
                    break;
            }
        }
        if (!err && src.width > 0 && src.height > 0) {
            // Optional inputs: neighbouring frames and the dirt texture,
            // each checked back in as soon as it is copied.
            glow_profond::GlowInputs inputs;
            Image prev, next, dirt;
            const PF_LRect* nb_rects[2] = {&plan->prev_rect, &plan->next_rect};
            Image* nb_imgs[2] = {&prev, &next};
            for (int i = 0; i < 2 && !err; ++i) {
                if (IsEmpty(*nb_rects[i])) continue;
                PF_EffectWorld* w = nullptr;
                ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, CHECKOUT_PREV + i, &w));
                if (!err && w && w->data) {
                    *nb_imgs[i] = Image(src.width, src.height, 4);
                    ERR(ReadWorld(format, w, *nb_imgs[i], int(nb_rects[i]->left - bounds.left),
                                  int(nb_rects[i]->top - bounds.top)));
                    if (!err) inputs.neighbors.push_back(nb_imgs[i]);
                }
                if (w) ERR2(extra->cb->checkin_layer_pixels(in_data->effect_ref, CHECKOUT_PREV + i));
            }
            if (!plan->depth) {
                // No depth layer: the depth controls do nothing.
                p.depth_spread = 0.0f;
                p.occlusion = false;
            }
            Image depth;
            if (!err && plan->depth) {
                PF_EffectWorld* w = nullptr;
                ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, CHECKOUT_DEPTH, &w));
                if (!err && w && w->data) {
                    Image tex(int(w->width), int(w->height), 4);
                    ERR(ReadWorld(format, w, tex, 0, 0));
                    if (!err) {
                        // The core reads red, straight, 0 = near. AE hands us
                        // premultiplied pixels; many Z passes are white = near.
                        // Transparent areas of the depth layer count as far.
                        for (size_t i = 0; i < tex.data.size(); i += 4) {
                            const float a = tex.data[i + 3];
                            float z = a > 0.0f ? tex.data[i] / a : 1.0f;
                            if (invert_depth && a > 0.0f) z = 1.0f - z;
                            tex.data[i] = tex.data[i + 1] = tex.data[i + 2] = z;
                            tex.data[i + 3] = 1.0f;
                        }
                        depth = ResampleLayer(tex, plan->layer_width, plan->layer_height, bounds.left, bounds.top,
                                             src.width, src.height);
                        inputs.depth = &depth;
                    }
                }
                if (w) ERR2(extra->cb->checkin_layer_pixels(in_data->effect_ref, CHECKOUT_DEPTH));
            }
            if (!err && plan->dirt) {
                PF_EffectWorld* w = nullptr;
                ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, CHECKOUT_DIRT, &w));
                if (!err && w && w->data) {
                    Image tex(int(w->width), int(w->height), 4);
                    ERR(ReadWorld(format, w, tex, 0, 0));
                    if (!err) {
                        dirt = ResampleLayer(tex, plan->layer_width, plan->layer_height, bounds.left, bounds.top,
                                            src.width, src.height);
                        inputs.dirt = &dirt;
                    }
                }
                if (w) ERR2(extra->cb->checkin_layer_pixels(in_data->effect_ref, CHECKOUT_DIRT));
            }
            if (!err) {
                const Image result = glow_profond::render_premultiplied(src, p, inputs);
                const A_long ox = plan->output_rect.left, oy = plan->output_rect.top;
                switch (format) {
                    case PF_PixelFormat_ARGB128:
                        WritePixels<PF_PixelFloat, PF_FpShort>(result, out_dx, out_dy, 1.0f, false, ox, oy, output);
                        break;
                    case PF_PixelFormat_ARGB64:
                        WritePixels<PF_Pixel16, A_u_short>(result, out_dx, out_dy, float(PF_MAX_CHAN16), true, ox, oy,
                                                           output);
                        break;
                    default:
                        WritePixels<PF_Pixel8, A_u_char>(result, out_dx, out_dy, float(PF_MAX_CHAN8), true, ox, oy, output);
                        break;
                }
            }
        }
    }

    if (input) ERR2(extra->cb->checkin_layer_pixels(in_data->effect_ref, GP_INPUT));
    return err;
}

#ifdef AE_OS_MAC
// Metal device setup: one GlowMetal (compiled kernels) per AE GPU device.
// Other frameworks (OpenCL, CUDA) are not set up, so AE stays on the CPU.
PF_Err GPUDeviceSetup(PF_InData* in_data, PF_OutData* out_data, PF_GPUDeviceSetupExtra* extra) {
    PF_Err err = PF_Err_NONE;
    if (extra->input->what_gpu != PF_GPU_Framework_METAL) return err;
    AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu(in_data, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_data);
    PF_GPUDeviceInfo info;
    AEFX_CLR_STRUCT(info);
    ERR(gpu->GetDeviceInfo(in_data->effect_ref, extra->input->device_index, &info));
    if (err || !info.devicePV) return err;
    std::string error;
    std::unique_ptr<glow_profond::metal::GlowMetal> metal =
        glow_profond::metal::GlowMetal::create_for_device(info.devicePV, &error);
    if (!metal) return err;  // no GPU flag: AE keeps rendering on the CPU
    extra->output->gpu_data = metal.release();
    out_data->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    return err;
}

PF_Err GPUDeviceSetdown(PF_GPUDeviceSetdownExtra* extra) {
    if (extra->input->what_gpu == PF_GPU_Framework_METAL) {
        delete static_cast<glow_profond::metal::GlowMetal*>(extra->input->gpu_data);
    }
    return PF_Err_NONE;
}

// Same rects and parameters as SmartRender, pixels stay on the GPU: AE's
// BGRA float worlds go straight into the Metal pipeline.
PF_Err SmartRenderGPU(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra) {
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    const RenderPlan* plan = static_cast<const RenderPlan*>(extra->input->pre_render_data);
    auto* metal = static_cast<glow_profond::metal::GlowMetal*>(const_cast<void*>(extra->input->gpu_data));
    if (!plan || !metal || extra->input->what_gpu != PF_GPU_Framework_METAL) return PF_Err_BAD_CALLBACK_PARAM;

    GlowParams p;
    ERR(ReadParams(in_data, out_data, p));
    PF_EffectWorld* input = nullptr;
    PF_EffectWorld* output = nullptr;
    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, GP_INPUT, &input));
    ERR(extra->cb->checkout_output(in_data->effect_ref, &output));

    if (!err && input && output) {
        AEFX_SuiteScoper<PF_WorldSuite2> worlds(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
        AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu(in_data, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_data);
        PF_PixelFormat format = PF_PixelFormat_INVALID;
        ERR(worlds->PF_GetPixelFormat(output, &format));
        if (!err && format != PF_PixelFormat_GPU_BGRA128) err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
        PF_GPUDeviceInfo info;
        AEFX_CLR_STRUCT(info);
        ERR(gpu->GetDeviceInfo(in_data->effect_ref, extra->input->device_index, &info));
        void* src_mem = nullptr;
        void* dst_mem = nullptr;
        ERR(gpu->GetGPUWorldData(in_data->effect_ref, input, &src_mem));
        ERR(gpu->GetGPUWorldData(in_data->effect_ref, output, &dst_mem));

        if (!err) {
            const PF_LRect bounds = WorkBounds(plan->input_rect, plan->output_rect, p);
            glow_profond::metal::GlowMetal::HostFrame src, dst;
            src.buffer = src_mem;
            src.width = int(input->width);
            src.height = int(input->height);
            src.pitch = int(input->rowbytes / 16);
            src.dx = int(plan->input_rect.left - bounds.left);
            src.dy = int(plan->input_rect.top - bounds.top);
            dst.buffer = dst_mem;
            dst.width = int(output->width);
            dst.height = int(output->height);
            dst.pitch = int(output->rowbytes / 16);
            dst.dx = int(plan->output_rect.left - bounds.left);
            dst.dy = int(plan->output_rect.top - bounds.top);
            const int work_w = int(bounds.right - bounds.left), work_h = int(bounds.bottom - bounds.top);
            // 8/16 bpc projects get the same colour <= alpha guard as on the CPU.
            const bool clamp_premult = extra->input->bitdepth < 32;
            if (!metal->submit_render_premultiplied(info.command_queuePV, src, dst, work_w, work_h, p, clamp_premult)) {
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }
        }
    }
    if (input) ERR2(extra->cb->checkin_layer_pixels(in_data->effect_ref, GP_INPUT));
    return err;
}
#endif

}  // namespace

extern "C" DllExport PF_Err PluginDataEntryFunction2(PF_PluginDataPtr inPtr, PF_PluginDataCB2 inPluginDataCallBackPtr,
                                                      SPBasicSuite* inSPBasicSuitePtr, const char* inHostName,
                                                      const char* inHostVersion) {
    // The SDK macro assigns to a local named `result`.
    PF_Err result = PF_Err_INVALID_CALLBACK;
    result = PF_REGISTER_EFFECT_EXT2(inPtr, inPluginDataCallBackPtr, GP_NAME, GP_MATCH_NAME, GP_CATEGORY,
                                     AE_RESERVED_INFO, "EffectMain", GP_SUPPORT_URL);
    return result;
}

extern "C" DllExport PF_Err EffectMain(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[],
                                       PF_LayerDef* output, void* extra) {
    (void)output;
    try {
        switch (cmd) {
            case PF_Cmd_ABOUT: return About(in_data, out_data);
            case PF_Cmd_GLOBAL_SETUP: return GlobalSetup(out_data);
            case PF_Cmd_PARAMS_SETUP: return ParamsSetup(in_data, out_data);
            case PF_Cmd_USER_CHANGED_PARAM:
                return UserChangedParam(params, static_cast<const PF_UserChangedParamExtra*>(extra));
            case PF_Cmd_UPDATE_PARAMS_UI: return UpdateParamsUI(in_data, out_data, params);
            case PF_Cmd_SMART_PRE_RENDER:
                return PreRender(in_data, out_data, static_cast<PF_PreRenderExtra*>(extra));
            case PF_Cmd_SMART_RENDER:
                return SmartRender(in_data, out_data, static_cast<PF_SmartRenderExtra*>(extra));
#ifdef AE_OS_MAC
            case PF_Cmd_GPU_DEVICE_SETUP:
                return GPUDeviceSetup(in_data, out_data, static_cast<PF_GPUDeviceSetupExtra*>(extra));
            case PF_Cmd_GPU_DEVICE_SETDOWN: return GPUDeviceSetdown(static_cast<PF_GPUDeviceSetdownExtra*>(extra));
            case PF_Cmd_SMART_RENDER_GPU:
                return SmartRenderGPU(in_data, out_data, static_cast<PF_SmartRenderExtra*>(extra));
#endif
            default: return PF_Err_NONE;
        }
    } catch (const std::bad_alloc&) {
        return PF_Err_OUT_OF_MEMORY;
    } catch (...) {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
}
