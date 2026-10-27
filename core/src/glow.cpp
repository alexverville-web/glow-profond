// Glow Profond: C++ port of reference/glow.py. Keep the two in step: every
// function here mirrors the reference function of the same name.
#include "glow_profond/glow.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <thread>

namespace glow_profond {
namespace {

// Variance (in source texels, per axis) that one 2x tent upsample adds:
// tent taps at +-1 (0.5) plus the bilinear split at +-0.25 texel (0.1875).
constexpr float kTentStepVariance = 0.6875f;

// Runs fn(row) for every row in [0, rows), split across hardware threads.
// Rows are handed out in small chunks from a shared counter: on Apple Silicon
// the efficiency cores are much slower, and a fixed split made every pass
// wait for them.
template <class Fn>
void parallel_rows(int rows, Fn fn) {
    const int hw = int(std::max(1u, std::thread::hardware_concurrency()));
    const int workers = std::min(hw, std::max(1, rows / 16));
    if (workers <= 1) {
        for (int y = 0; y < rows; ++y) fn(y);
        return;
    }
    const int chunk = std::max(1, rows / (workers * 8));
    std::atomic<int> next{0};
    auto work = [&] {
        for (int start; (start = next.fetch_add(chunk)) < rows;) {
            const int end = std::min(rows, start + chunk);
            for (int y = start; y < end; ++y) fn(y);
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(size_t(workers - 1));
    for (int w = 1; w < workers; ++w) pool.emplace_back(work);
    work();  // the calling thread works too
    for (auto& t : pool) t.join();
}

// sRGB curves tabulated on [0, 1] against sqrt(x): in that variable both are
// smooth enough for linear interpolation to stay under 1e-6, where pow() was
// the most expensive per-pixel operation. Values above 1 (HDR) use pow().
constexpr int kSrgbLutSize = 4096;

struct SrgbLuts {
    float to_linear[kSrgbLutSize + 1];
    float to_srgb[kSrgbLutSize + 1];
    SrgbLuts() {
        for (int i = 0; i <= kSrgbLutSize; ++i) {
            const double t = double(i) / kSrgbLutSize, x = t * t;
            to_linear[i] = float(x <= 0.04045 ? x / 12.92 : std::pow((x + 0.055) / 1.055, 2.4));
            to_srgb[i] = float(x <= 0.0031308 ? x * 12.92 : 1.055 * std::pow(x, 1.0 / 2.4) - 0.055);
        }
    }
};

const SrgbLuts& srgb_luts() {
    static const SrgbLuts luts;  // thread-safe one-time init
    return luts;
}

float lut_lookup(const float* table, float x) {
    const float f = std::sqrt(x) * float(kSrgbLutSize);
    const int i = std::min(int(f), kSrgbLutSize - 1);
    return table[i] + (table[i + 1] - table[i]) * (f - float(i));
}

std::vector<float> gauss_kernel(float sigma) {
    const int r = std::max(1, int(std::ceil(3.0f * sigma)));
    const float s = std::max(sigma, 1e-3f);
    std::vector<float> k(static_cast<size_t>(2 * r + 1));
    float sum = 0.0f;
    for (int i = -r; i <= r; ++i) {
        const float x = float(i) / s;
        k[size_t(i + r)] = std::exp(-0.5f * x * x);
        sum += k[size_t(i + r)];
    }
    for (float& v : k) v /= sum;
    return k;
}

Rgb tint_for_level(const std::vector<Rgb>& tint, int k, int n) {
    if (tint.empty()) return {1.0f, 1.0f, 1.0f};
    if (tint.size() == 1) return tint[0];
    const float t = float(k) / float(std::max(n - 1, 1)) * float(tint.size() - 1);
    const int i = std::min(int(t), int(tint.size()) - 2);
    const float f = t - float(i);
    Rgb c;
    for (int ch = 0; ch < 3; ++ch) c[ch] = tint[i][ch] * (1.0f - f) + tint[i + 1][ch] * f;
    return c;
}

}  // namespace

float srgb_to_linear(float x) {
    x = std::max(x, 0.0f);
    if (x <= 1.0f) return lut_lookup(srgb_luts().to_linear, x);
    return x <= 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f);
}

float linear_to_srgb(float x) {
    x = std::max(x, 0.0f);
    if (x <= 1.0f) return lut_lookup(srgb_luts().to_srgb, x);
    return x <= 0.0031308f ? x * 12.92f : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}

float aces_fitted(float x) {
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return std::clamp(x * (a * x + b) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

Image soft_threshold(const Image& rgb, float t, float smooth) {
    Image out = rgb;
    // Knee relative to the threshold: t = 0 passes the image through unchanged.
    const float knee = std::max(t * smooth, 1e-5f);
    parallel_rows(rgb.height, [&](int y) {
        for (int x = 0; x < rgb.width; ++x) {
            float* p = out.px(x, y);
            const float br = std::max({p[0], p[1], p[2]});
            const float s = std::clamp(br - t + knee, 0.0f, 2.0f * knee);
            const float soft = s * s / (4.0f * knee);
            const float contrib = std::max(soft, br - t) / std::max(br, 1e-6f);
            for (int c = 0; c < 3; ++c) p[c] *= contrib;
        }
    });
    return out;
}

Image downsample2(const Image& img) {
    // Odd sizes round up, with zeros past the edge (like the reference):
    // texel i always covers source pixels 2i and 2i + 1, so every level stays
    // aligned with the image whatever the buffer size.
    Image out((img.width + 1) / 2, (img.height + 1) / 2, img.channels);
    const int ch = img.channels;
    parallel_rows(out.height, [&](int y) {
        const bool has_y1 = 2 * y + 1 < img.height;
        for (int x = 0; x < out.width; ++x) {
            const bool has_x1 = 2 * x + 1 < img.width;
            const float* a = img.px(2 * x, 2 * y);
            const float* b = has_x1 ? img.px(2 * x + 1, 2 * y) : nullptr;
            const float* c = has_y1 ? img.px(2 * x, 2 * y + 1) : nullptr;
            const float* d = has_x1 && has_y1 ? img.px(2 * x + 1, 2 * y + 1) : nullptr;
            float* o = out.px(x, y);
            for (int k = 0; k < ch; ++k) {
                // Same summation order as the reference: a + c + b + d.
                o[k] = 0.25f * (a[k] + (c ? c[k] : 0.0f) + (b ? b[k] : 0.0f) + (d ? d[k] : 0.0f));
            }
        }
    });
    return out;
}

namespace {

// Bilinear resize; source coordinate of target pixel i is (i + 0.5) * ratio - 0.5.
Image resample_bilinear(const Image& img, int width, int height, float ratio_x, float ratio_y) {
    Image out(width, height, img.channels);
    const int w = img.width, h = img.height;
    std::vector<int> x0(static_cast<size_t>(width)), x1(static_cast<size_t>(width));
    std::vector<float> fx(static_cast<size_t>(width));
    for (int x = 0; x < width; ++x) {
        const float xs = std::clamp((float(x) + 0.5f) * ratio_x - 0.5f, 0.0f, float(w - 1));
        x0[x] = int(std::floor(xs));
        x1[x] = std::min(x0[x] + 1, w - 1);
        fx[x] = xs - float(x0[x]);
    }
    parallel_rows(height, [&](int y) {
        const float ys = std::clamp((float(y) + 0.5f) * ratio_y - 0.5f, 0.0f, float(h - 1));
        const int y0 = int(std::floor(ys));
        const int y1 = std::min(y0 + 1, h - 1);
        const float fy = ys - float(y0);
        for (int x = 0; x < width; ++x) {
            const float* p00 = img.px(x0[x], y0);
            const float* p01 = img.px(x1[x], y0);
            const float* p10 = img.px(x0[x], y1);
            const float* p11 = img.px(x1[x], y1);
            float* o = out.px(x, y);
            for (int c = 0; c < img.channels; ++c) {
                const float top = p00[c] * (1.0f - fx[x]) + p01[c] * fx[x];
                const float bot = p10[c] * (1.0f - fx[x]) + p11[c] * fx[x];
                o[c] = top * (1.0f - fy) + bot * fy;
            }
        }
    });
    return out;
}

}  // namespace

Image upsample_to(const Image& img, int width, int height) {
    return resample_bilinear(img, width, height, float(img.width) / float(width),
                             float(img.height) / float(height));
}

Image upsample_by(const Image& img, int width, int height, int factor) {
    return resample_bilinear(img, width, height, 1.0f / float(factor), 1.0f / float(factor));
}

void blur_axis(Image& img, int channel, float sigma, bool vertical) {
    if (sigma < 0.3f) return;
    const std::vector<float> k = gauss_kernel(sigma);
    const int r = int(k.size()) / 2;
    const int n = vertical ? img.height : img.width;
    const int lines = vertical ? img.width : img.height;
    const int ch = img.channels;
    std::vector<float> src_all(img.data);
    parallel_rows(lines, [&](int line) {
        for (int i = 0; i < n; ++i) {
            float acc = 0.0f;
            // Zero padding outside the image, like np.pad(mode="constant").
            const int lo = std::max(0, i - r), hi = std::min(n - 1, i + r);
            for (int j = lo; j <= hi; ++j) {
                const size_t idx = vertical ? (size_t(j) * img.width + line) : (size_t(line) * img.width + j);
                acc += k[size_t(j - i + r)] * src_all[idx * ch + channel];
            }
            const size_t o = vertical ? (size_t(i) * img.width + line) : (size_t(line) * img.width + i);
            img.data[o * ch + channel] = acc;
        }
    });
}

namespace {

// Separable Gaussian on every channel of a level, both axes, with one
// scratch copy per axis. Same arithmetic as blur_axis (zero padding, same
// summation order), so results match it bit for bit.
void blur_level(Image& img, const std::vector<float>& sx, const std::vector<float>& sy) {
    const int W = img.width, H = img.height, ch = img.channels;
    std::vector<std::vector<float>> kx(static_cast<size_t>(ch)), ky(static_cast<size_t>(ch));
    bool any_x = false, any_y = false;
    for (int c = 0; c < ch; ++c) {
        if (sx[c] >= 0.3f) { kx[c] = gauss_kernel(sx[c]); any_x = true; }
        if (sy[c] >= 0.3f) { ky[c] = gauss_kernel(sy[c]); any_y = true; }
    }
    // Without chromatic aberration all channels share one kernel: then each
    // tap is a multiply-add over a whole contiguous row, which vectorizes.
    const bool same_x = std::all_of(sx.begin(), sx.end(), [&](float s) { return s == sx[0]; });
    const bool same_y = std::all_of(sy.begin(), sy.end(), [&](float s) { return s == sy[0]; });
    const size_t stride = size_t(W) * ch;
    if (any_x) {
        int rmax = 0;
        for (int c = 0; c < ch; ++c) rmax = std::max(rmax, int(kx[c].size()) / 2);
        const std::vector<float> src(img.data);
        parallel_rows(H, [&](int y) {
            // Zero-padded copy of the row: taps never need bounds checks, and
            // adding k * 0 leaves the sums identical to blur_axis.
            std::vector<float> pad(stride + size_t(2 * rmax) * ch, 0.0f);
            std::copy_n(src.data() + size_t(y) * stride, stride, pad.data() + size_t(rmax) * ch);
            float* out = img.data.data() + size_t(y) * stride;
            if (same_x) {
                const int r = int(kx[0].size()) / 2;
                std::fill(out, out + stride, 0.0f);
                for (int j = -r; j <= r; ++j) {
                    const float kj = kx[0][size_t(j + r)];
                    const float* in = pad.data() + size_t(rmax + j) * ch;
                    for (size_t i = 0; i < stride; ++i) out[i] += kj * in[i];
                }
                return;
            }
            for (int c = 0; c < ch; ++c) {
                if (kx[c].empty()) continue;
                const int r = int(kx[c].size()) / 2;
                for (int x = 0; x < W; ++x) out[size_t(x) * ch + c] = 0.0f;
                for (int j = -r; j <= r; ++j) {
                    const float kj = kx[c][size_t(j + r)];
                    const float* in = pad.data() + size_t(rmax + j) * ch + c;
                    for (int x = 0; x < W; ++x) out[size_t(x) * ch + c] += kj * in[size_t(x) * ch];
                }
            }
        });
    }
    if (any_y) {
        const std::vector<float> src(img.data);
        // Row-major traversal: output row i gathers whole input rows, which
        // keeps memory access sequential (columns would thrash the cache).
        parallel_rows(H, [&](int i) {
            float* out = img.data.data() + size_t(i) * stride;
            if (same_y) {
                const int r = int(ky[0].size()) / 2;
                const int lo = std::max(0, i - r), hi = std::min(H - 1, i + r);
                std::fill(out, out + stride, 0.0f);
                for (int j = lo; j <= hi; ++j) {
                    const float kj = ky[0][size_t(j - i + r)];
                    const float* row = src.data() + size_t(j) * stride;
                    for (size_t x = 0; x < stride; ++x) out[x] += kj * row[x];
                }
                return;
            }
            for (int c = 0; c < ch; ++c) {
                if (ky[c].empty()) continue;
                const float* k = ky[c].data();
                const int r = int(ky[c].size()) / 2;
                const int lo = std::max(0, i - r), hi = std::min(H - 1, i + r);
                for (int x = 0; x < W; ++x) out[size_t(x) * ch + c] = 0.0f;
                for (int j = lo; j <= hi; ++j) {
                    const float kj = k[j - i + r];
                    const float* row = src.data() + size_t(j) * stride;
                    for (int x = 0; x < W; ++x) out[size_t(x) * ch + c] += kj * row[size_t(x) * ch + c];
                }
            }
        });
    }
}

}  // namespace

namespace {

// One output coordinate of upsample_tent2: the three bilinear taps folded
// into at most four distinct source texels with summed weights.
struct TentTaps {
    int idx[4];
    float w[4];
    int n;
};

std::vector<TentTaps> tent_taps(int target, int n) {
    static const float kTent[3] = {0.25f, 0.5f, 0.25f};
    std::vector<TentTaps> t(static_cast<size_t>(target));
    for (int i = 0; i < target; ++i) {
        TentTaps& tt = t[size_t(i)];
        tt.n = 0;
        auto add = [&](int idx, float w) {
            for (int m = 0; m < tt.n; ++m) {
                if (tt.idx[m] == idx) {
                    tt.w[m] += w;
                    return;
                }
            }
            tt.idx[tt.n] = idx;
            tt.w[tt.n++] = w;
        };
        // Taps at -1, 0, +1 source texels around the target's position in
        // source coordinates (exactly 2x, the inverse of downsample2),
        // clamped like upsample_to, each read bilinearly.
        const float base = (float(i) + 0.5f) * 0.5f - 0.5f;
        for (int d = 0; d < 3; ++d) {
            const float c = std::clamp(base + float(d - 1), 0.0f, float(n - 1));
            const int i0 = int(std::floor(c));
            const float f = c - float(i0);
            add(i0, kTent[d] * (1.0f - f));
            add(std::min(i0 + 1, n - 1), kTent[d] * f);
        }
    }
    return t;
}

// upsample_tent2(img) + add, in one pass (add may be null): the cascade adds
// each level right after upsampling the coarser sum into it.
Image upsample_tent2_add(const Image& img, int width, int height, const Image* add) {
    const int ch = img.channels;
    const auto tx = tent_taps(width, img.width);
    const auto ty = tent_taps(height, img.height);

    // Along x first (source rows, target width), then along y.
    Image rows(width, img.height, ch);
    parallel_rows(img.height, [&](int y) {
        for (int x = 0; x < width; ++x) {
            const TentTaps& t = tx[size_t(x)];
            float* o = rows.px(x, y);
            for (int c = 0; c < ch; ++c) o[c] = 0.0f;
            for (int m = 0; m < t.n; ++m) {
                const float* s = img.px(t.idx[m], y);
                for (int c = 0; c < ch; ++c) o[c] += t.w[m] * s[c];
            }
        }
    });
    // Along y every tap is a whole row: contiguous multiply-adds.
    Image out(width, height, ch);
    const size_t stride = size_t(width) * ch;
    parallel_rows(height, [&](int y) {
        const TentTaps& t = ty[size_t(y)];
        float* o = out.px(0, y);
        if (add) {
            std::copy_n(add->px(0, y), stride, o);
        } else {
            std::fill(o, o + stride, 0.0f);
        }
        for (int m = 0; m < t.n; ++m) {
            const float w = t.w[m];
            const float* r = rows.px(0, t.idx[m]);
            for (size_t i = 0; i < stride; ++i) o[i] += w * r[i];
        }
    });
    return out;
}

}  // namespace

Image upsample_tent2(const Image& img, int width, int height) {
    return upsample_tent2_add(img, width, height, nullptr);
}

void parallel_for_rows(int rows, const std::function<void(int)>& fn) { parallel_rows(rows, fn); }

LevelPlan level_plan(const GlowParams& p) {
    // Levels at base_sigma * 2^k, except the widest, whose sigma is exactly
    // the radius and whose weight fades in with the fractional octave: the
    // halo grows smoothly with the radius instead of jumping a level at a time.
    // Weights are not normalized, so a larger radius only adds halo further out.
    const double nf = std::log2(std::max(double(p.radius) / double(p.base_sigma), 1.0)) + 1.0;
    const int n = std::max(1, int(std::ceil(nf - 1e-6)));
    LevelPlan plan;
    plan.sigmas.resize(static_cast<size_t>(n));
    plan.weights.resize(static_cast<size_t>(n));
    for (int k = 0; k < n; ++k) {
        const double sigma = double(p.base_sigma) * std::pow(2.0, k == n - 1 ? nf - 1.0 : double(k));
        double w = std::pow(sigma / double(p.base_sigma), 2.0 - double(p.spread));
        if (k == n - 1) w *= std::min(nf - double(n - 1), 1.0);
        plan.sigmas[size_t(k)] = float(sigma);
        plan.weights[size_t(k)] = float(w);
    }
    return plan;
}

int level_count(const GlowParams& p) { return int(level_plan(p).sigmas.size()); }

namespace {

float smoothstep(float e0, float e1, float x) {
    const float t = std::clamp((x - e0) / std::max(e1 - e0, 1e-6f), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Saturated sources glow towards white in the core; scaling the chroma
// around the Rec. 709 luma lets a neon keep (or lose) its colour.
void saturate(Image& glow, float s) {
    if (s == 1.0f) return;
    parallel_rows(glow.height, [&](int y) {
        float* a = glow.px(0, y);
        for (int x = 0; x < glow.width; ++x, a += 3) {
            const float luma = 0.2126f * a[0] + 0.7152f * a[1] + 0.0722f * a[2];
            for (int c = 0; c < 3; ++c) a[c] = std::max(luma + s * (a[c] - luma), 0.0f);
        }
    });
}

// First channel of img as a 1-channel map of size W x H (bilinear if needed).
Image depth_map(const Image& img, int W, int H) {
    Image d(img.width, img.height, 1);
    for (size_t i = 0; i < d.data.size(); ++i) d.data[i] = img.data[i * size_t(img.channels)];
    return (d.width == W && d.height == H) ? d : upsample_to(d, W, H);
}

}  // namespace

Image glow_layer(const Image& src, const GlowParams& p, const Image* depth, const Image* dirt) {
    const int H = src.height, W = src.width;
    Image thr = soft_threshold(src, p.threshold, p.threshold_smooth);
    Image d;
    if (depth) d = depth_map(*depth, W, H);
    const int extra = int(std::log2(float(std::max(p.downsample, 1))));
    for (int i = 0; i < extra; ++i) {
        thr = downsample2(thr);
        if (depth) d = downsample2(d);
    }

    if (thr.width == 0 || thr.height == 0) return Image(W, H, 3);

    const float scale = float(1 << extra);
    const LevelPlan plan = level_plan(p);
    const std::vector<float>& weights = plan.weights;
    const int n = int(weights.size());
    float wsum = 0.0f;
    for (float w : weights) wsum += w;

    const float ax = std::sqrt(p.aspect), ay = 1.0f / std::sqrt(p.aspect);
    const bool occlude = p.occlusion && depth;
    const float spread = depth ? std::clamp(p.depth_spread, 0.0f, 1.0f) : 0.0f;
    const bool want_dirt = dirt && p.dirt_amount > 0.0f;
    const int large_from = n / 2;  // levels feeding the lens dirt

    // Depth spread: a source at depth d sends w_k * lerp(1, 2^(k - n/2), spread * d)
    // to level k, renormalized so its total is 1. That factor is affine in d
    // once divided by Z(d), so two pyramids (a, b) replace one per level:
    // level input k = down^k(a) + c_k * down^k(b), c_k = 2^(k - n/2) - 1.
    std::vector<float> c(static_cast<size_t>(n));
    float wc = 0.0f;
    for (int k = 0; k < n; ++k) {
        c[k] = std::pow(2.0f, float(k) - float(n) / 2.0f) - 1.0f;
        wc += weights[k] / wsum * c[k];
    }
    // Channels 3-4 carry intensity and intensity * depth for occlusion.
    const int pc = occlude ? 5 : 3;
    auto extend = [&](Image& out, int x, int y, const float* rgb) {
        float* o = out.px(x, y);
        for (int ch = 0; ch < 3; ++ch) o[ch] = rgb[ch];
        if (occlude) {
            const float i = (rgb[0] + rgb[1] + rgb[2]) / 3.0f;
            o[3] = i;
            o[4] = i * d.px(x, y)[0];
        }
    };
    std::vector<Image> pyr_a, pyr_b;
    if (pc == 3 && spread == 0.0f) {
        pyr_a.push_back(std::move(thr));  // plain glow: the thresholded image is level 0
    } else {
        Image a(thr.width, thr.height, pc), b;
        if (spread > 0.0f) b = Image(thr.width, thr.height, pc);
        parallel_rows(thr.height, [&](int y) {
            for (int x = 0; x < thr.width; ++x) {
                const float* t = thr.px(x, y);
                if (spread > 0.0f) {
                    const float sd = spread * d.px(x, y)[0];
                    const float z = 1.0f + sd * wc;
                    const float va[3] = {t[0] / z, t[1] / z, t[2] / z};
                    const float vb[3] = {t[0] * sd / z, t[1] * sd / z, t[2] * sd / z};
                    extend(a, x, y, va);
                    extend(b, x, y, vb);
                } else {
                    extend(a, x, y, t);
                }
            }
        });
        pyr_a.push_back(std::move(a));
        if (spread > 0.0f) pyr_b.push_back(std::move(b));
    }
    for (int k = 1; k < n; ++k) {
        Image next = downsample2(pyr_a.back());
        if (next.width == 0 || next.height == 0) break;  // tiny images: fewer levels
        pyr_a.push_back(std::move(next));
        if (!pyr_b.empty()) pyr_b.push_back(downsample2(pyr_b.back()));
    }
    const int used = int(pyr_a.size());

    // Folded channels: tinted glow (3), untinted wide glow for the dirt (3),
    // occlusion sums (2). Only what the settings need is carried.
    const int large_off = 3, occ_off = want_dirt ? 6 : 3;
    const int ac = occ_off + (occlude ? 2 : 0);
    // Exposure is folded into the level weights: one less full-size pass
    // (the occlusion ratio is unaffected by a common gain).
    const float exposure_gain = std::pow(2.0f, p.exposure);

    // Level k blurred and weighted, with the ac folded channels. The blur is
    // shrunk by the variance its upsample chain adds back, so the effective
    // sigma stays base_sigma * 2^k.
    auto contribution = [&](int k) {
        Image lvl = std::move(pyr_a[size_t(k)]);
        if (!pyr_b.empty()) {
            const Image& b = pyr_b[size_t(k)];
            for (size_t i = 0; i < lvl.data.size(); ++i) lvl.data[i] += c[k] * b.data[i];
        }
        const float var = kTentStepVariance * (4.0f / 3.0f) * (1.0f - std::pow(4.0f, -float(k)));
        std::vector<float> sx(static_cast<size_t>(pc)), sy(static_cast<size_t>(pc));
        for (int ch = 0; ch < pc; ++ch) {
            // Sigma in level k's own pixels: base_sigma for all but the widest.
            const float sigma_level = plan.sigmas[size_t(k)] / (std::pow(2.0f, float(k)) * scale);
            const float s = sigma_level * (ch < 3 ? p.rgb_radius[ch] : 1.0f);
            sx[ch] = std::sqrt(std::max((s * ax) * (s * ax) - var, 0.0f));
            sy[ch] = std::sqrt(std::max((s * ay) * (s * ay) - var, 0.0f));
        }
        blur_level(lvl, sx, sy);
        const float w = weights[k] * exposure_gain;
        const Rgb tint = tint_for_level(p.tint, k, n);
        Rgb gain;
        for (int ch = 0; ch < 3; ++ch) gain[ch] = w * tint[ch];
        const float large = k >= large_from ? w : 0.0f;
        Image out = ac == pc ? std::move(lvl) : Image(lvl.width, lvl.height, ac);
        const Image& in = ac == pc ? out : lvl;
        if (ac == 3) {  // plain glow, in place
            parallel_rows(out.height, [&](int y) {
                float* v = out.px(0, y);
                for (int x = 0; x < out.width; ++x, v += 3)
                    for (int ch = 0; ch < 3; ++ch) v[ch] *= gain[ch];
            });
            return out;
        }
        parallel_rows(out.height, [&](int y) {
            for (int x = 0; x < out.width; ++x) {
                const float* v = in.px(x, y);
                float* o = out.px(x, y);
                const float rgb[3] = {v[0], v[1], v[2]};
                const float occ[2] = {occlude ? v[3] : 0.0f, occlude ? v[4] : 0.0f};
                for (int ch = 0; ch < 3; ++ch) o[ch] = gain[ch] * rgb[ch];
                if (want_dirt)
                    for (int ch = 0; ch < 3; ++ch) o[large_off + ch] = large * rgb[ch];
                if (occlude) {
                    o[occ_off] = w * occ[0];
                    o[occ_off + 1] = w * occ[1];
                }
            }
        });
        return out;
    };

    Image acc = contribution(used - 1);
    for (int k = used - 2; k >= 0; --k) {
        const Image lvl = contribution(k);
        acc = upsample_tent2_add(acc, lvl.width, lvl.height, &lvl);
    }
    if (acc.width != W || acc.height != H) acc = upsample_by(acc, W, H, 1 << extra);
    if (ac == 3) {
        saturate(acc, p.saturation);
        return acc;
    }

    Image d_full, dirt_full;
    if (occlude) d_full = depth_map(*depth, W, H);
    if (want_dirt) {
        dirt_full = (dirt->width == W && dirt->height == H) ? Image() : upsample_to(*dirt, W, H);
    }
    const Image* dirt_img = want_dirt ? (dirt_full.data.empty() ? dirt : &dirt_full) : nullptr;
    Image glow(W, H, 3);
    parallel_rows(H, [&](int y) {
        for (int x = 0; x < W; ++x) {
            const float* a = acc.px(x, y);
            float* g = glow.px(x, y);
            float keep = 1.0f;
            if (occlude) {
                const float d_mean = a[occ_off + 1] / std::max(a[occ_off], 1e-8f);
                keep = 1.0f - smoothstep(0.0f, p.occlusion_softness, d_mean - d_full.px(x, y)[0]);
            }
            for (int ch = 0; ch < 3; ++ch) g[ch] = a[ch] * keep;
            if (dirt_img) {
                const float* dt = dirt_img->px(x, y);
                for (int ch = 0; ch < 3; ++ch) g[ch] += p.dirt_amount * dt[ch] * a[large_off + ch];
            }
        }
    });
    saturate(glow, p.saturation);
    return glow;
}

namespace {

// Screen is a + b - ab on channels at or below 1.
float composite(float base, float glow, Blend blend) {
    return blend == Blend::Screen ? base + glow * (1.0f - std::clamp(base, 0.0f, 1.0f)) : base + glow;
}

}  // namespace

Image render(const Image& src, const GlowParams& p, const Image* depth, const Image* dirt) {
    const int W = src.width, H = src.height;
    Image lin(W, H, 3);
    std::vector<float> coverage(size_t(W) * H);
    parallel_rows(H, [&](int y) {
        for (int x = 0; x < W; ++x) {
            const float* s = src.px(x, y);
            float rgb[3] = {s[0], s[1], s[2]};
            float alpha = src.channels == 4 ? s[3] : 1.0f;
            if (p.unmult) {
                alpha = std::clamp(std::max({rgb[0], rgb[1], rgb[2]}), 0.0f, 1.0f);
                for (float& v : rgb) v /= std::max(alpha, 1e-6f);
            }
            float* o = lin.px(x, y);
            for (int c = 0; c < 3; ++c) o[c] = (p.gamma_correct ? srgb_to_linear(rgb[c]) : rgb[c]) * alpha;
            coverage[size_t(y) * W + x] = alpha;
        }
    });

    Image out = lin;
    if (p.view != View::Source) {
        Image dirt_lin;
        if (dirt && p.dirt_amount > 0.0f) {
            dirt_lin = Image(dirt->width, dirt->height, 3);
            for (int y = 0; y < dirt->height; ++y)
                for (int x = 0; x < dirt->width; ++x)
                    for (int c = 0; c < 3; ++c) {
                        const float v = dirt->px(x, y)[c];
                        dirt_lin.px(x, y)[c] = p.gamma_correct ? srgb_to_linear(v) : v;
                    }
        }
        const Image g = glow_layer(lin, p, depth, dirt_lin.data.empty() ? nullptr : &dirt_lin);
        if (p.view == View::Glow) {
            out = g;
        } else {
            for (size_t i = 0; i < out.data.size(); ++i) {
                // Behind: the glow passes under the layer, through its transparency.
                const float under = p.view == View::Behind ? 1.0f - coverage[i / 3] : 1.0f;
                out.data[i] = composite(out.data[i], under * g.data[i], p.blend);
            }
        }
    }
    parallel_rows(H, [&](int y) {
        float* row = out.px(0, y);
        for (int i = 0; i < W * 3; ++i) {
            if (p.tonemap) row[i] = aces_fitted(row[i]);
            if (p.gamma_correct) row[i] = linear_to_srgb(row[i]);
        }
    });
    return out;
}

Image render_premultiplied(const Image& src, const GlowParams& p, const GlowInputs& inputs) {
    const int W = src.width, H = src.height;
    Image lin(W, H, 3);
    std::vector<float> alpha(size_t(W) * H);
    parallel_rows(H, [&](int y) {
        for (int x = 0; x < W; ++x) {
            const float* s = src.px(x, y);
            float a = s[3];
            float rgb[3] = {0.0f, 0.0f, 0.0f};
            if (a > 0.0f) {
                for (int c = 0; c < 3; ++c) rgb[c] = s[c] / a;  // straight colour
            }
            if (p.unmult) {
                const float m = std::clamp(std::max({rgb[0], rgb[1], rgb[2]}), 0.0f, 1.0f);
                for (float& v : rgb) v /= std::max(m, 1e-6f);
                a *= m;
            }
            alpha[size_t(y) * W + x] = a;
            float* o = lin.px(x, y);
            for (int c = 0; c < 3; ++c) o[c] = (p.gamma_correct ? srgb_to_linear(rgb[c]) : rgb[c]) * a;
        }
    });

    // Glow source: the current frame, optionally blended with its
    // neighbours so a highlight that flickers frame to frame glows steadily.
    auto linearize = [&](const Image& img, bool premultiplied) {
        Image l(W, H, 3);
        parallel_rows(H, [&](int y) {
            for (int x = 0; x < W; ++x) {
                const float* s = img.px(x, y);
                float a = img.channels == 4 ? s[3] : 1.0f;
                float* o = l.px(x, y);
                for (int c = 0; c < 3; ++c) {
                    float v = s[c];
                    if (premultiplied) v = a > 0.0f ? v / a : 0.0f;
                    if (p.gamma_correct) v = srgb_to_linear(v);
                    o[c] = v * a;
                }
            }
        });
        return l;
    };

    Image glow;
    if (p.view != View::Source) {
        const float t = std::clamp(p.temporal_smooth, 0.0f, 1.0f);
        const Image* glow_src = &lin;
        Image blended;
        if (t > 0.0f && !inputs.neighbors.empty()) {
            blended = lin;
            const float keep = 1.0f - 0.5f * t;
            const float each = 0.5f * t / float(inputs.neighbors.size());
            for (float& v : blended.data) v *= keep;
            for (const Image* nb : inputs.neighbors) {
                const Image l = linearize(*nb, true);
                for (size_t i = 0; i < blended.data.size(); ++i) blended.data[i] += each * l.data[i];
            }
            glow_src = &blended;
        }
        Image dirt_lin;
        if (inputs.dirt && p.dirt_amount > 0.0f) dirt_lin = linearize(*inputs.dirt, false);
        // Depth is data, not colour: read straight, never linearized.
        Image depth;
        if (inputs.depth) {
            const Image& d = *inputs.depth;
            depth = Image(d.width, d.height, 1);
            for (int y = 0; y < d.height; ++y)
                for (int x = 0; x < d.width; ++x) {
                    const float* s = d.px(x, y);
                    const float a = d.channels == 4 ? s[3] : 1.0f;
                    depth.px(x, y)[0] = d.channels == 4 ? (a > 0.0f ? s[0] / a : 1.0f) : s[0];
                }
        }
        glow = glow_layer(*glow_src, p, inputs.depth ? &depth : nullptr,
                          dirt_lin.data.empty() ? nullptr : &dirt_lin);
    }

    // Linear light -> output encoding (tonemap, then sRGB when gamma_correct).
    auto encode = [&](float v) {
        if (p.tonemap) v = aces_fitted(v);
        if (p.gamma_correct) v = linear_to_srgb(v);
        return v;
    };
    Image out(W, H, 4);
    parallel_rows(H, [&](int y) {
        for (int x = 0; x < W; ++x) {
            const float* l = lin.px(x, y);
            float* o = out.px(x, y);
            const float a = alpha[size_t(y) * W + x];
            // The source keeps its usual encoding: straight colour encoded,
            // premultiplied again. Semi-transparent edges are unchanged.
            float src_enc[3];
            for (int c = 0; c < 3; ++c) src_enc[c] = a > 0.0f ? encode(l[c] / a) * a : 0.0f;
            if (p.view == View::Source) {
                for (int c = 0; c < 3; ++c) o[c] = src_enc[c];
                o[3] = a;
                continue;
            }
            // The glow is light added over the layer, so it is encoded as it
            // looks over black: encode(source + glow) - encode(source). The
            // old version kept it linear and put its linear value in alpha,
            // so a wide halo (linear 0.005) came out at 1/255: invisible.
            const float* g = glow.px(x, y);
            const bool glow_only = p.view == View::Glow;
            // Behind: the glow passes under the layer, through its transparency.
            const float under = p.view == View::Behind ? 1.0f - a : 1.0f;
            float rgb[3], peak = 0.0f;
            for (int c = 0; c < 3; ++c) {
                const float base = glow_only ? 0.0f : l[c];
                const float total = glow_only ? g[c] : composite(l[c], under * g[c], p.blend);
                const float added = std::max(encode(total) - encode(base), 0.0f);
                rgb[c] = (glow_only ? 0.0f : src_enc[c]) + added;
                peak = std::max(peak, rgb[c]);
            }
            // Coverage: enough alpha for the colour to stay premultiplied
            // (colour <= alpha up to 1), so over black it shows exactly rgb.
            const float base_alpha = glow_only ? 0.0f : a;
            for (int c = 0; c < 3; ++c) o[c] = rgb[c];
            o[3] = std::max(base_alpha, std::min(peak, 1.0f));
        }
    });
    return out;
}

}  // namespace glow_profond
