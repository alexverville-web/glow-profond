// Glow Profond: Metal kernels for glow_layer (see core/src/glow.cpp).
//
// Every kernel mirrors one CPU building block, in the same summation order,
// so GPU and CPU agree to float rounding. Images are float4 buffers (RGB +
// unused A) of w * h pixels, row-major. Compiled at runtime from source:
// the Command Line Tools have no offline Metal compiler.
#include <metal_stdlib>
using namespace metal;

struct Dims {
    uint w, h;
};

// soft_threshold: knee relative to the threshold (t = 0 passes through).
kernel void soft_threshold(device const float4* src [[buffer(0)]], device float4* dst [[buffer(1)]],
                           constant Dims& d [[buffer(2)]], constant float2& tk [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= d.w || gid.y >= d.h) return;
    const uint i = gid.y * d.w + gid.x;
    const float3 p = src[i].xyz;
    const float t = tk.x, knee = tk.y;
    const float br = max(max(p.x, p.y), p.z);
    const float s = clamp(br - t + knee, 0.0f, 2.0f * knee);
    const float soft = s * s / (4.0f * knee);
    const float contrib = max(soft, br - t) / max(br, 1e-6f);
    dst[i] = float4(p * contrib, 0.0f);
}

// downsample2: [1 3 3 1] / 8 per axis (anti-aliased); odd sizes round up
// (dst is ceil(w/2) x ceil(h/2)) with zeros past the edge, so texel i is
// centred on pixels 2i and 2i + 1. Same order as the core: along x within
// each of the four rows, then across the rows.
static float4 tap4(float4 a, float4 b, float4 c, float4 d) { return (a + 3.0f * b + 3.0f * c + d) * 0.125f; }

kernel void downsample2(device const float4* src [[buffer(0)]], device float4* dst [[buffer(1)]],
                        constant Dims& sd [[buffer(2)]], constant Dims& dd [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dd.w || gid.y >= dd.h) return;
    const int x = 2 * int(gid.x), y = 2 * int(gid.y), w = int(sd.w), h = int(sd.h);
    float4 rows[4];
    for (int j = 0; j < 4; ++j) {
        const int yy = y - 1 + j;
        float4 t[4];
        for (int i = 0; i < 4; ++i) {
            const int xx = x - 1 + i;
            t[i] = (yy >= 0 && yy < h && xx >= 0 && xx < w) ? src[uint(yy) * sd.w + uint(xx)] : float4(0.0f);
        }
        rows[j] = tap4(t[0], t[1], t[2], t[3]);
    }
    dst[gid.y * dd.w + gid.x] = tap4(rows[0], rows[1], rows[2], rows[3]);
}

// Per-channel Gaussian taps for one axis: channel c uses weights
// k[off[c] .. off[c] + 2 r[c]]; r = 0 is the identity (sigma below 0.3).
struct BlurParams {
    int4 r;
    int4 off;
    float4 gain;  // applied after the vertical pass: level weight * tint
};

kernel void blur_h(device const float4* src [[buffer(0)]], device float4* dst [[buffer(1)]],
                   constant Dims& d [[buffer(2)]], constant float* k [[buffer(3)]],
                   constant BlurParams& bp [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= d.w || gid.y >= d.h) return;
    const int x = int(gid.x), w = int(d.w);
    device const float4* row = src + gid.y * d.w;
    float acc[3];
    for (int c = 0; c < 3; ++c) {
        const int r = bp.r[c];
        const int lo = max(0, x - r), hi = min(w - 1, x + r);
        float a = 0.0f;
        for (int j = lo; j <= hi; ++j) a += k[bp.off[c] + j - x + r] * row[j][c];
        acc[c] = a;
    }
    dst[gid.y * d.w + gid.x] = float4(acc[0], acc[1], acc[2], 0.0f);
}

kernel void blur_v(device const float4* src [[buffer(0)]], device float4* dst [[buffer(1)]],
                   constant Dims& d [[buffer(2)]], constant float* k [[buffer(3)]],
                   constant BlurParams& bp [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= d.w || gid.y >= d.h) return;
    const int y = int(gid.y), h = int(d.h);
    float acc[3];
    for (int c = 0; c < 3; ++c) {
        const int r = bp.r[c];
        const int lo = max(0, y - r), hi = min(h - 1, y + r);
        float a = 0.0f;
        for (int j = lo; j <= hi; ++j) a += k[bp.off[c] + j - y + r] * src[uint(j) * d.w + gid.x][c];
        acc[c] = a * bp.gain[c];
    }
    dst[gid.y * d.w + gid.x] = float4(acc[0], acc[1], acc[2], 0.0f);
}

// upsample_tent2, one output coordinate: up to four source texels with
// their summed weights (precomputed on the CPU exactly like the core).
struct TentTaps {
    int4 idx;
    float4 w;
    int n;
    int pad[3];
};

// Along x: source rows -> target width (dst is tw x sh).
kernel void tent_x(device const float4* src [[buffer(0)]], device float4* dst [[buffer(1)]],
                   constant Dims& sd [[buffer(2)]], constant Dims& dd [[buffer(3)]],
                   constant TentTaps* taps [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dd.w || gid.y >= dd.h) return;
    const TentTaps t = taps[gid.x];
    device const float4* row = src + gid.y * sd.w;
    float4 o = 0.0f;
    for (int m = 0; m < t.n; ++m) o += t.w[m] * row[t.idx[m]];
    dst[gid.y * dd.w + gid.x] = o;
}

// Along y, plus the finer level: dst = add + sum_m w_m * rows[idx_m].
kernel void tent_y_add(device const float4* rows [[buffer(0)]], device float4* dst [[buffer(1)]],
                       constant Dims& dd [[buffer(2)]], device const float4* add [[buffer(3)]],
                       constant TentTaps* taps [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dd.w || gid.y >= dd.h) return;
    const TentTaps t = taps[gid.y];
    const uint i = gid.y * dd.w + gid.x;
    float4 o = add[i];
    for (int m = 0; m < t.n; ++m) o += t.w[m] * rows[uint(t.idx[m]) * dd.w + gid.x];
    dst[i] = o;
}

// Bilinear read at output pixel (x, y) of img (sd) upsampled by exactly
// 1 / ratio (upsample_by in the core: identity when ratio is 1).
static float4 bilinear_at(device const float4* img, Dims sd, float ratio, uint x, uint y) {
    const float w = float(sd.w), h = float(sd.h);
    const float xs = clamp((float(x) + 0.5f) * ratio - 0.5f, 0.0f, w - 1.0f);
    const float ys = clamp((float(y) + 0.5f) * ratio - 0.5f, 0.0f, h - 1.0f);
    const uint x0 = uint(floor(xs)), y0 = uint(floor(ys));
    const uint x1 = min(x0 + 1, sd.w - 1), y1 = min(y0 + 1, sd.h - 1);
    const float fx = xs - float(x0), fy = ys - float(y0);
    const float4 top = img[y0 * sd.w + x0] * (1.0f - fx) + img[y0 * sd.w + x1] * fx;
    const float4 bot = img[y1 * sd.w + x0] * (1.0f - fx) + img[y1 * sd.w + x1] * fx;
    return top * (1.0f - fy) + bot * fy;
}

struct FinishParams {
    Dims acc;     // cascade result (full size, or smaller after a global downsample)
    Dims out;
    float ratio;  // 1 / 2^downsample steps
    float saturation;
    int use_saturation;
    int pad;
};

// End of glow_layer: back to full size (identity when sizes match), then
// saturation around the Rec. 709 luma.
kernel void finish(device const float4* acc [[buffer(0)]], device float4* dst [[buffer(1)]],
                   constant FinishParams& fp [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= fp.out.w || gid.y >= fp.out.h) return;
    float3 a = bilinear_at(acc, fp.acc, fp.ratio, gid.x, gid.y).xyz;
    if (fp.use_saturation) {
        const float luma = 0.2126f * a.x + 0.7152f * a.y + 0.0722f * a.z;
        a = max(luma + fp.saturation * (a - luma), 0.0f);
    }
    dst[gid.y * fp.out.w + gid.x] = float4(a, 0.0f);
}

// ---------------------------------------------------------------------------
// Full render_premultiplied on host frames (After Effects GPU worlds): BGRA
// float4, premultiplied, rows `pitch` pixels apart. See core/src/glow.cpp.

static float srgb_to_linear(float x) {
    x = max(x, 0.0f);
    return x <= 0.04045f ? x / 12.92f : pow((x + 0.055f) / 1.055f, 2.4f);
}

static float linear_to_srgb(float x) {
    x = max(x, 0.0f);
    return x <= 0.0031308f ? x * 12.92f : 1.055f * pow(x, 1.0f / 2.4f) - 0.055f;
}

static float aces_fitted(float x) {
    return clamp(x * (2.51f * x + 0.03f) / (x * (2.43f * x + 0.59f) + 0.14f), 0.0f, 1.0f);
}

struct PrepareParams {
    uint src_w, src_h, src_pitch;
    int dx, dy;  // where the source frame sits in the work buffer
    uint work_w, work_h;
    int unmult, gamma, clamp_premult;
};

// Host frame -> work buffer: linear premultiplied RGB, alpha in w; black
// where the source doesn't cover the work area.
kernel void prepare(device const float4* src [[buffer(0)]], device float4* work [[buffer(1)]],
                    constant PrepareParams& pp [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= pp.work_w || gid.y >= pp.work_h) return;
    const int sx = int(gid.x) - pp.dx, sy = int(gid.y) - pp.dy;
    float4 o = 0.0f;
    if (sx >= 0 && sy >= 0 && sx < int(pp.src_w) && sy < int(pp.src_h)) {
        const float4 px = src[uint(sy) * pp.src_pitch + uint(sx)];  // B, G, R, A
        float a = px.w;
        float3 c = float3(px.z, px.y, px.x);
        // 8/16-bit projects: colour <= alpha (see ReadPixels in the AE wrapper).
        if (pp.clamp_premult) c = min(c, float3(a));
        float3 rgb = a > 0.0f ? c / a : float3(0.0f);
        if (pp.unmult) {
            const float m = clamp(max(max(rgb.x, rgb.y), rgb.z), 0.0f, 1.0f);
            rgb /= max(m, 1e-6f);
            a *= m;
        }
        if (pp.gamma) rgb = float3(srgb_to_linear(rgb.x), srgb_to_linear(rgb.y), srgb_to_linear(rgb.z));
        o = float4(rgb * a, a);
    }
    work[gid.y * pp.work_w + gid.x] = o;
}

struct CompositeParams {
    uint dst_w, dst_h, dst_pitch;
    int dx, dy;  // where the output frame sits in the work buffer
    uint work_w;
    int view;    // 0 final, 1 glow only, 2 source, 3 behind
    int screen, gamma, tonemap;
};

static float encode(float v, constant CompositeParams& cp) {
    if (cp.tonemap) v = aces_fitted(v);
    if (cp.gamma) v = linear_to_srgb(v);
    return v;
}

// Source + glow -> host frame, encoded as it looks over black, alpha large
// enough to stay premultiplied (same rule as render_premultiplied).
kernel void composite(device const float4* work [[buffer(0)]], device const float4* glow [[buffer(1)]],
                      device float4* dst [[buffer(2)]], constant CompositeParams& cp [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= cp.dst_w || gid.y >= cp.dst_h) return;
    const uint i = uint(int(gid.y) + cp.dy) * cp.work_w + uint(int(gid.x) + cp.dx);
    const float4 l = work[i];
    const float a = l.w;
    float src_enc[3], rgb[3];
    for (int c = 0; c < 3; ++c) src_enc[c] = a > 0.0f ? encode(l[c] / a, cp) * a : 0.0f;
    float alpha = a;
    if (cp.view == 2) {
        for (int c = 0; c < 3; ++c) rgb[c] = src_enc[c];
    } else {
        const float3 g = glow[i].xyz;
        const bool glow_only = cp.view == 1;
        const float under = cp.view == 3 ? 1.0f - a : 1.0f;
        float peak = 0.0f;
        for (int c = 0; c < 3; ++c) {
            const float base = glow_only ? 0.0f : l[c];
            const float add = under * g[c];
            const float total = glow_only ? g[c]
                                          : (cp.screen ? l[c] + add * (1.0f - clamp(l[c], 0.0f, 1.0f)) : l[c] + add);
            const float added = max(encode(total, cp) - encode(base, cp), 0.0f);
            rgb[c] = (glow_only ? 0.0f : src_enc[c]) + added;
            peak = max(peak, rgb[c]);
        }
        alpha = max(glow_only ? 0.0f : a, min(peak, 1.0f));
    }
    dst[gid.y * cp.dst_pitch + gid.x] = float4(rgb[2], rgb[1], rgb[0], alpha);
}
