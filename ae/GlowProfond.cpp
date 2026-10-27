// Glow Profond: After Effects SmartFX wrapper around the glow core.
//
// The look lives in core/ (a port of reference/glow.py); this file only maps
// AE parameters and pixel buffers onto glow_profond::render_premultiplied.
// 8, 16 and 32 bpc are supported; buffers are converted to float RGBA.
#include <algorithm>
#include <cmath>
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

static_assert(GP_OUT_FLAGS == (PF_OutFlag_DEEP_COLOR_AWARE | PF_OutFlag_I_EXPAND_BUFFER),
              "PiPL out flags out of sync with GlobalSetup");
static_assert(GP_OUT_FLAGS_2 == (PF_OutFlag2_FLOAT_COLOR_AWARE | PF_OutFlag2_SUPPORTS_SMART_RENDER |
                                 PF_OutFlag2_SUPPORTS_THREADED_RENDERING |
                                 PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG),
              "PiPL out flags 2 out of sync with GlobalSetup");
static_assert(GP_VERSION_PACKED == PF_VERSION(GP_MAJOR_VERSION, GP_MINOR_VERSION, GP_BUG_VERSION,
                                              GP_STAGE_VERSION, GP_BUILD_VERSION),
              "PiPL version out of sync with my_version");

namespace {

using glow_profond::GlowParams;
using glow_profond::Image;

// Parameter order in the Effect Controls panel.
enum {
    GP_INPUT = 0,
    GP_RADIUS,
    GP_EXPOSURE,
    GP_SPREAD,
    GP_INPUT_GROUP,
    GP_THRESHOLD,
    GP_THRESHOLD_SMOOTH,
    GP_UNMULT,
    GP_INPUT_GROUP_END,
    GP_SHAPE_GROUP,
    GP_ASPECT,
    GP_RADIUS_R,
    GP_RADIUS_G,
    GP_RADIUS_B,
    GP_SHAPE_GROUP_END,
    GP_TINT_GROUP,
    GP_TINT_MODE,
    GP_TINT_CORE,
    GP_TINT_MID,
    GP_TINT_HALO,
    GP_TINT_GROUP_END,
    GP_RENDER_GROUP,
    GP_GAMMA,
    GP_TONEMAP,
    GP_BASE_SIGMA,
    GP_DOWNSAMPLE,
    GP_RENDER_GROUP_END,
    GP_VIEW,
    GP_NUM_PARAMS
};

// Disk IDs are saved in projects: never renumber, only append.
enum {
    ID_RADIUS = 1,
    ID_EXPOSURE,
    ID_SPREAD,
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
    ID_VIEW
};

enum { TINT_NONE = 1, TINT_TWO, TINT_THREE };
enum { VIEW_FINAL = 1, VIEW_GLOW, VIEW_SOURCE };

// What PreRender decided, handed to SmartRender.
struct RenderPlan {
    PF_LRect input_rect;   // layer-space rect covered by the input world
    PF_LRect output_rect;  // layer-space rect covered by the output world
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

float ToFloat(const PF_ParamDef& def) { return static_cast<float>(def.u.fs_d.value); }

// Reads every parameter at the current time. Pixel sizes are scaled to the
// render resolution, so a half-resolution preview matches the full render.
PF_Err ReadParams(PF_InData* in_data, PF_OutData* out_data, GlowParams& p) {
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    static const int kValueParams[] = {GP_RADIUS,    GP_EXPOSURE,  GP_SPREAD,     GP_THRESHOLD,
                                       GP_THRESHOLD_SMOOTH, GP_UNMULT, GP_ASPECT, GP_RADIUS_R,
                                       GP_RADIUS_G,  GP_RADIUS_B,  GP_TINT_MODE,  GP_TINT_CORE,
                                       GP_TINT_MID,  GP_TINT_HALO, GP_GAMMA,      GP_TONEMAP,
                                       GP_BASE_SIGMA, GP_DOWNSAMPLE, GP_VIEW};
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
        p.radius = ToFloat(defs[GP_RADIUS]) * scale;
        p.base_sigma = ToFloat(defs[GP_BASE_SIGMA]) * scale;
        p.exposure = ToFloat(defs[GP_EXPOSURE]);
        p.spread = ToFloat(defs[GP_SPREAD]);
        p.threshold = ToFloat(defs[GP_THRESHOLD]);
        p.threshold_smooth = ToFloat(defs[GP_THRESHOLD_SMOOTH]);
        p.unmult = defs[GP_UNMULT].u.bd.value != 0;
        p.aspect = ToFloat(defs[GP_ASPECT]);
        p.rgb_radius = {ToFloat(defs[GP_RADIUS_R]), ToFloat(defs[GP_RADIUS_G]), ToFloat(defs[GP_RADIUS_B])};
        p.gamma_correct = defs[GP_GAMMA].u.bd.value != 0;
        p.tonemap = defs[GP_TONEMAP].u.bd.value != 0;
        p.downsample = 1 << std::clamp<A_long>(defs[GP_DOWNSAMPLE].u.pd.value - 1, 0, 3);

        switch (defs[GP_VIEW].u.pd.value) {
            case VIEW_GLOW: p.view = glow_profond::View::Glow; break;
            case VIEW_SOURCE: p.view = glow_profond::View::Source; break;
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

// How far the glow can reach beyond its source, in pixels: three sigmas of
// the widest level, stretched by aspect and per-channel radius.
A_long GlowPadding(const GlowParams& p) {
    if (p.view == glow_profond::View::Source) return 0;
    const float max_mult = std::max({p.rgb_radius[0], p.rgb_radius[1], p.rgb_radius[2], 1.0f});
    const float stretch = std::max(std::sqrt(p.aspect), 1.0f / std::sqrt(p.aspect));
    const float reach = 3.0f * p.radius * max_mult * stretch + 4.0f * p.downsample;
    return static_cast<A_long>(std::ceil(std::min(reach, 30000.0f)));
}

template <class Pixel>
void ReadPixels(const PF_EffectWorld* world, float max_value, Image& img, int dx, int dy) {
    for (A_long y = 0; y < world->height; ++y) {
        const Pixel* row = reinterpret_cast<const Pixel*>(static_cast<const char*>(world->data) + y * world->rowbytes);
        for (A_long x = 0; x < world->width; ++x) {
            float* o = img.px(int(x) + dx, int(y) + dy);
            o[0] = float(row[x].red) / max_value;
            o[1] = float(row[x].green) / max_value;
            o[2] = float(row[x].blue) / max_value;
            o[3] = float(row[x].alpha) / max_value;
        }
    }
}

template <class Pixel, class Channel>
void WritePixels(const Image& img, int dx, int dy, float max_value, bool clamp, PF_EffectWorld* world) {
    auto conv = [&](float v) {
        if (clamp) return static_cast<Channel>(std::clamp(v, 0.0f, 1.0f) * max_value + 0.5f);
        return static_cast<Channel>(v * max_value);
    };
    for (A_long y = 0; y < world->height; ++y) {
        Pixel* row = reinterpret_cast<Pixel*>(static_cast<char*>(world->data) + y * world->rowbytes);
        for (A_long x = 0; x < world->width; ++x) {
            const float* s = img.px(int(x) + dx, int(y) + dy);
            row[x].red = conv(s[0]);
            row[x].green = conv(s[1]);
            row[x].blue = conv(s[2]);
            row[x].alpha = conv(s[3]);
        }
    }
}

PF_Err About(PF_InData* in_data, PF_OutData* out_data) {
    PF_SPRINTF(out_data->return_msg, "%s v%d.%d.%d\rGimmick Studio\rGlow a falloff physique (1/r^p).", GP_NAME,
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

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Rayon", 0, 4000, 0, 1000, 400, PF_Precision_TENTHS, 0, 0, ID_RADIUS);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Exposition", -10, 10, -4, 4, 0, PF_Precision_HUNDREDTHS, 0, 0, ID_EXPOSURE);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Diffusion (falloff)", 0.5, 4, 1, 3, 2, PF_Precision_HUNDREDTHS, 0, 0, ID_SPREAD);

    AEFX_CLR_STRUCT(def);
    PF_ADD_TOPIC("Entree", ID_INPUT_GROUP);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Seuil", 0, 100, 0, 2, 0, PF_Precision_HUNDREDTHS, 0, 0, ID_THRESHOLD);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Douceur du seuil", 0, 10, 0, 1, 0.25, PF_Precision_HUNDREDTHS, 0, 0, ID_THRESHOLD_SMOOTH);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Unmult (alpha = luma)", FALSE, 0, ID_UNMULT);
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(ID_INPUT_GROUP_END);

    AEFX_CLR_STRUCT(def);
    PF_ADD_TOPIC("Forme", ID_SHAPE_GROUP);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Ratio d'aspect", 0.1, 10, 0.25, 4, 1, PF_Precision_HUNDREDTHS, 0, 0, ID_ASPECT);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Rayon rouge", 0, 4, 0.5, 2, 1, PF_Precision_HUNDREDTHS, 0, 0, ID_RADIUS_R);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Rayon vert", 0, 4, 0.5, 2, 1, PF_Precision_HUNDREDTHS, 0, 0, ID_RADIUS_G);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Rayon bleu", 0, 4, 0.5, 2, 1, PF_Precision_HUNDREDTHS, 0, 0, ID_RADIUS_B);
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(ID_SHAPE_GROUP_END);

    AEFX_CLR_STRUCT(def);
    PF_ADD_TOPIC("Teinte", ID_TINT_GROUP);
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Mode de teinte", 3, TINT_NONE, "Aucune|2 couleurs|3 couleurs", ID_TINT_MODE);
    AEFX_CLR_STRUCT(def);
    PF_ADD_COLOR("Couleur du coeur", 255, 255, 255, ID_TINT_CORE);
    AEFX_CLR_STRUCT(def);
    PF_ADD_COLOR("Couleur du milieu", 255, 102, 204, ID_TINT_MID);
    AEFX_CLR_STRUCT(def);
    PF_ADD_COLOR("Couleur du halo", 77, 77, 255, ID_TINT_HALO);
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(ID_TINT_GROUP_END);

    AEFX_CLR_STRUCT(def);
    PF_ADD_TOPIC("Rendu", ID_RENDER_GROUP);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Correction gamma", TRUE, 0, ID_GAMMA);
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Tonemap ACES", FALSE, 0, ID_TONEMAP);
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Sigma de base", 0.5, 8, 0.5, 4, 1.5, PF_Precision_HUNDREDTHS, 0, 0, ID_BASE_SIGMA);
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Sous-echantillonnage", 4, 1, "1x|2x|4x|8x", ID_DOWNSAMPLE);
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(ID_RENDER_GROUP_END);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Affichage", 3, VIEW_FINAL, "Resultat final|Glow seul|Source", ID_VIEW);

    out_data->num_params = GP_NUM_PARAMS;
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
    extra->output->pre_render_data = plan;
    extra->output->delete_pre_render_data_func = DeleteRenderPlan;
    return err;
}

PF_Err SmartRender(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra) {
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    const RenderPlan* plan = static_cast<const RenderPlan*>(extra->input->pre_render_data);
    if (!plan) return PF_Err_BAD_CALLBACK_PARAM;

    GlowParams p;
    ERR(ReadParams(in_data, out_data, p));

    PF_EffectWorld* input = nullptr;
    PF_EffectWorld* output = nullptr;
    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, GP_INPUT, &input));
    ERR(extra->cb->checkout_output(in_data->effect_ref, &output));

    if (!err && input && output) {
        AEFX_SuiteScoper<PF_WorldSuite2> worlds(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
        PF_PixelFormat format = PF_PixelFormat_INVALID;
        ERR(worlds->PF_GetPixelFormat(output, &format));

        // Work on the union of input and output so the glow sees its sources.
        const PF_LRect bounds = Union(plan->input_rect, plan->output_rect);
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
            const Image result = glow_profond::render_premultiplied(src, p);
            switch (format) {
                case PF_PixelFormat_ARGB128:
                    WritePixels<PF_PixelFloat, PF_FpShort>(result, out_dx, out_dy, 1.0f, false, output);
                    break;
                case PF_PixelFormat_ARGB64:
                    WritePixels<PF_Pixel16, A_u_short>(result, out_dx, out_dy, float(PF_MAX_CHAN16), true, output);
                    break;
                default:
                    WritePixels<PF_Pixel8, A_u_char>(result, out_dx, out_dy, float(PF_MAX_CHAN8), true, output);
                    break;
            }
        }
    }

    if (input) ERR2(extra->cb->checkin_layer_pixels(in_data->effect_ref, GP_INPUT));
    return err;
}

}  // namespace

extern "C" DllExport PF_Err PluginDataEntryFunction2(PF_PluginDataPtr inPtr, PF_PluginDataCB2 inPluginDataCallBackPtr,
                                                      SPBasicSuite* inSPBasicSuitePtr, const char* inHostName,
                                                      const char* inHostVersion) {
    return PF_REGISTER_EFFECT_EXT2(inPtr, inPluginDataCallBackPtr, GP_NAME, GP_MATCH_NAME, GP_CATEGORY,
                                   AE_RESERVED_INFO, "EffectMain", GP_SUPPORT_URL);
}

extern "C" DllExport PF_Err EffectMain(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[],
                                       PF_LayerDef* output, void* extra) {
    (void)params;
    (void)output;
    try {
        switch (cmd) {
            case PF_Cmd_ABOUT: return About(in_data, out_data);
            case PF_Cmd_GLOBAL_SETUP: return GlobalSetup(out_data);
            case PF_Cmd_PARAMS_SETUP: return ParamsSetup(in_data, out_data);
            case PF_Cmd_SMART_PRE_RENDER:
                return PreRender(in_data, out_data, static_cast<PF_PreRenderExtra*>(extra));
            case PF_Cmd_SMART_RENDER:
                return SmartRender(in_data, out_data, static_cast<PF_SmartRenderExtra*>(extra));
            default: return PF_Err_NONE;
        }
    } catch (const std::bad_alloc&) {
        return PF_Err_OUT_OF_MEMORY;
    } catch (...) {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
}
