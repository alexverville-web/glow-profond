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

// downsample2: 2x2 box, odd sizes truncated (dst is floor(w/2) x floor(h/2)).
kernel void downsample2(device const float4* src [[buffer(0)]], device float4* dst [[buffer(1)]],
                        constant Dims& sd [[buffer(2)]], constant Dims& dd [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dd.w || gid.y >= dd.h) return;
    const uint r0 = (2 * gid.y) * sd.w, r1 = r0 + sd.w, x = 2 * gid.x;
    const float4 a = src[r0 + x], b = src[r0 + x + 1], c = src[r1 + x], e = src[r1 + x + 1];
    dst[gid.y * dd.w + gid.x] = 0.25f * (a + c + b + e);
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

// Bilinear read at the position of output pixel (x, y) when img (sd) is
// stretched over od, in accumulate_upsampled's arithmetic order.
static float4 bilinear_at(device const float4* img, Dims sd, Dims od, uint x, uint y) {
    const float w = float(sd.w), h = float(sd.h);
    const float xs = clamp((float(x) + 0.5f) * w / float(od.w) - 0.5f, 0.0f, w - 1.0f);
    const float ys = clamp((float(y) + 0.5f) * h / float(od.h) - 0.5f, 0.0f, h - 1.0f);
    const uint x0 = uint(floor(xs)), y0 = uint(floor(ys));
    const uint x1 = min(x0 + 1, sd.w - 1), y1 = min(y0 + 1, sd.h - 1);
    const float fx = xs - float(x0), fy = ys - float(y0);
    const float4 top = img[y0 * sd.w + x0] * (1.0f - fx) + img[y0 * sd.w + x1] * fx;
    const float4 bot = img[y1 * sd.w + x0] * (1.0f - fx) + img[y1 * sd.w + x1] * fx;
    return top * (1.0f - fy) + bot * fy;
}

struct FinishParams {
    Dims acc;  // cascade result (full size, or smaller after a global downsample)
    Dims out;
    float saturation;
    int use_saturation;
};

// End of glow_layer: back to full size (identity when sizes match), then
// saturation around the Rec. 709 luma.
kernel void finish(device const float4* acc [[buffer(0)]], device float4* dst [[buffer(1)]],
                   constant FinishParams& fp [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= fp.out.w || gid.y >= fp.out.h) return;
    float3 a = bilinear_at(acc, fp.acc, fp.out, gid.x, gid.y).xyz;
    if (fp.use_saturation) {
        const float luma = 0.2126f * a.x + 0.7152f * a.y + 0.0722f * a.z;
        a = max(luma + fp.saturation * (a - luma), 0.0f);
    }
    dst[gid.y * fp.out.w + gid.x] = float4(a, 0.0f);
}
