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
    // Odd dimensions are truncated, like the reference.
    Image out(img.width / 2, img.height / 2, img.channels);
    parallel_rows(out.height, [&](int y) {
        for (int x = 0; x < out.width; ++x) {
            const float* a = img.px(2 * x, 2 * y);
            const float* b = img.px(2 * x + 1, 2 * y);
            const float* c = img.px(2 * x, 2 * y + 1);
            const float* d = img.px(2 * x + 1, 2 * y + 1);
            float* o = out.px(x, y);
            for (int ch = 0; ch < img.channels; ++ch) o[ch] = 0.25f * (a[ch] + c[ch] + b[ch] + d[ch]);
        }
    });
    return out;
}

Image upsample_to(const Image& img, int width, int height) {
    Image out(width, height, img.channels);
    const int w = img.width, h = img.height;
    std::vector<int> x0(static_cast<size_t>(width)), x1(static_cast<size_t>(width));
    std::vector<float> fx(static_cast<size_t>(width));
    for (int x = 0; x < width; ++x) {
        const float xs = std::clamp((float(x) + 0.5f) * float(w) / float(width) - 0.5f, 0.0f, float(w - 1));
        x0[x] = int(std::floor(xs));
        x1[x] = std::min(x0[x] + 1, w - 1);
        fx[x] = xs - float(x0[x]);
    }
    parallel_rows(height, [&](int y) {
        const float ys = std::clamp((float(y) + 0.5f) * float(h) / float(height) - 0.5f, 0.0f, float(h - 1));
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

// Separable Gaussian on the three channels of a level, both axes, with one
// scratch copy per axis. Same arithmetic as blur_axis (zero padding, same
// summation order), so results match it bit for bit.
void blur_level(Image& img, const float (&sx)[3], const float (&sy)[3]) {
    const int W = img.width, H = img.height, ch = img.channels;
    std::vector<float> kx[3], ky[3];
    bool any_x = false, any_y = false;
    for (int c = 0; c < 3; ++c) {
        if (sx[c] >= 0.3f) { kx[c] = gauss_kernel(sx[c]); any_x = true; }
        if (sy[c] >= 0.3f) { ky[c] = gauss_kernel(sy[c]); any_y = true; }
    }
    // Without chromatic aberration all channels share one kernel: then each
    // tap is a multiply-add over a whole contiguous row, which vectorizes.
    const bool same_x = ch == 3 && sx[0] == sx[1] && sx[1] == sx[2];
    const bool same_y = ch == 3 && sy[0] == sy[1] && sy[1] == sy[2];
    const size_t stride = size_t(W) * ch;
    if (any_x) {
        int rmax = 0;
        for (int c = 0; c < 3; ++c) rmax = std::max(rmax, int(kx[c].size()) / 2);
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
            for (int c = 0; c < 3; ++c) {
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
            for (int c = 0; c < 3; ++c) {
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

// acc += gain * upsample_to(img, acc size), without materializing the
// upsampled image. Same interpolation arithmetic as upsample_to.
void accumulate_upsampled(Image& acc, const Image& img, const Rgb& gain) {
    const int width = acc.width, height = acc.height;
    const int w = img.width, h = img.height;
    std::vector<int> x0(static_cast<size_t>(width)), x1(static_cast<size_t>(width));
    std::vector<float> fx(static_cast<size_t>(width));
    for (int x = 0; x < width; ++x) {
        const float xs = std::clamp((float(x) + 0.5f) * float(w) / float(width) - 0.5f, 0.0f, float(w - 1));
        x0[x] = int(std::floor(xs));
        x1[x] = std::min(x0[x] + 1, w - 1);
        fx[x] = xs - float(x0[x]);
    }
    parallel_rows(height, [&](int y) {
        const float ys = std::clamp((float(y) + 0.5f) * float(h) / float(height) - 0.5f, 0.0f, float(h - 1));
        const int y0 = int(std::floor(ys));
        const int y1 = std::min(y0 + 1, h - 1);
        const float fy = ys - float(y0);
        for (int x = 0; x < width; ++x) {
            const float* p00 = img.px(x0[x], y0);
            const float* p01 = img.px(x1[x], y0);
            const float* p10 = img.px(x0[x], y1);
            const float* p11 = img.px(x1[x], y1);
            float* a = acc.px(x, y);
            for (int c = 0; c < 3; ++c) {
                const float top = p00[c] * (1.0f - fx[x]) + p01[c] * fx[x];
                const float bot = p10[c] * (1.0f - fx[x]) + p11[c] * fx[x];
                a[c] += gain[c] * (top * (1.0f - fy) + bot * fy);
            }
        }
    });
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
        // source coordinates, clamped like upsample_to, each read bilinearly.
        const float base = (float(i) + 0.5f) * float(n) / float(target) - 0.5f;
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

int level_count(const GlowParams& p) {
    const float ratio = std::max(p.radius / p.base_sigma, 1.0f);
    return std::max(1, int(std::floor(std::log2(ratio))) + 1);
}

Image glow_layer(const Image& src, const GlowParams& p, const Image* dirt) {
    const int H = src.height, W = src.width;
    Image img = soft_threshold(src, p.threshold, p.threshold_smooth);
    const int extra = int(std::log2(float(std::max(p.downsample, 1))));
    for (int i = 0; i < extra; ++i) img = downsample2(img);

    if (img.width == 0 || img.height == 0) return Image(W, H, 3);

    const float scale = float(H) / float(img.height);
    const int n = level_count(p);
    std::vector<float> weights(static_cast<size_t>(n));
    float wsum = 0.0f;
    for (int k = 0; k < n; ++k) {
        weights[k] = std::pow(std::pow(2.0f, float(k)), 2.0f - p.spread);
        wsum += weights[k];
    }
    // Exposure is folded into the level weights: one less full-size pass.
    const float exposure_gain = std::pow(2.0f, p.exposure);
    for (float& w : weights) w *= exposure_gain / wsum;

    const float ax = std::sqrt(p.aspect), ay = 1.0f / std::sqrt(p.aspect);
    const float sigma_level = p.base_sigma / scale;  // same in every level's own pixels
    std::vector<Image> levels;
    levels.push_back(std::move(img));
    for (int k = 1; k < n; ++k) {
        Image next = downsample2(levels.back());
        if (next.width == 0 || next.height == 0) break;  // tiny images: fewer levels
        levels.push_back(std::move(next));
    }
    const int used = int(levels.size());

    // Blur level k in place and scale it by its weight and tint. The blur is
    // shrunk by the variance its upsample chain adds back, so the effective
    // sigma stays base_sigma * 2^k.
    auto blur_weighted = [&](int k) {
        const float extra = kTentStepVariance * (4.0f / 3.0f) * (1.0f - std::pow(4.0f, -float(k)));
        float sx[3], sy[3];
        for (int c = 0; c < 3; ++c) {
            const float s = sigma_level * p.rgb_radius[c];
            sx[c] = std::sqrt(std::max((s * ax) * (s * ax) - extra, 0.0f));
            sy[c] = std::sqrt(std::max((s * ay) * (s * ay) - extra, 0.0f));
        }
        Image& lvl = levels[size_t(k)];
        blur_level(lvl, sx, sy);
        const Rgb tint = tint_for_level(p.tint, k, n);
        Rgb gain;
        for (int c = 0; c < 3; ++c) gain[c] = weights[k] * tint[c];
        parallel_rows(lvl.height, [&](int y) {
            for (int x = 0; x < lvl.width; ++x) {
                float* v = lvl.px(x, y);
                for (int c = 0; c < 3; ++c) v[c] = gain[c] * v[c];
            }
        });
    };

    // Lens dirt is lit by the wide half of the pyramid: the cascade holds
    // exactly that sum once it has folded down to level used / 2.
    const bool want_dirt = dirt && p.dirt_amount > 0.0f;
    const int dirt_level = used / 2;
    Image wide;

    blur_weighted(used - 1);
    Image cascade = std::move(levels[size_t(used - 1)]);
    if (want_dirt && dirt_level == used - 1) wide = cascade;
    for (int k = used - 2; k >= 0; --k) {
        blur_weighted(k);
        const Image& lvl = levels[size_t(k)];
        cascade = upsample_tent2_add(cascade, lvl.width, lvl.height, &lvl);
        if (want_dirt && k == dirt_level) wide = cascade;
    }
    Image acc;
    if (cascade.width == W && cascade.height == H) {
        acc = std::move(cascade);
    } else {
        acc = Image(W, H, 3);
        accumulate_upsampled(acc, cascade, {1.0f, 1.0f, 1.0f});
    }
    if (p.saturation != 1.0f) {
        // Saturated sources glow towards white in the core; scaling the
        // chroma around the luma lets a neon keep (or lose) its colour.
        const float s = p.saturation;
        parallel_rows(H, [&](int y) {
            float* a = acc.px(0, y);
            for (int x = 0; x < W; ++x, a += 3) {
                const float luma = 0.2126f * a[0] + 0.7152f * a[1] + 0.0722f * a[2];
                for (int c = 0; c < 3; ++c) a[c] = std::max(luma + s * (a[c] - luma), 0.0f);
            }
        });
    }
    if (want_dirt) {
        Image lit(W, H, 3);
        accumulate_upsampled(lit, wide, {p.dirt_amount, p.dirt_amount, p.dirt_amount});
        parallel_rows(H, [&](int y) {
            for (int x = 0; x < W; ++x) {
                const float* d = dirt->px(x, y);
                const float* l = lit.px(x, y);
                float* a = acc.px(x, y);
                for (int c = 0; c < 3; ++c) a[c] += d[c] * l[c];
            }
        });
    }
    return acc;
}

Image render(const Image& src, const GlowParams& p) {
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
        const Image g = glow_layer(lin, p);
        if (p.view == View::Glow) {
            out = g;
        } else {
            for (size_t i = 0; i < out.data.size(); ++i) {
                const float base = out.data[i];
                float add = p.blend == Blend::Screen
                                ? std::min(g.data[i], 1.0f) * (1.0f - std::clamp(base, 0.0f, 1.0f))
                                : g.data[i];
                if (p.view == View::Behind) add *= 1.0f - coverage[i / 3];  // only where the layer isn't
                out.data[i] += add;
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
        glow = glow_layer(*glow_src, p, dirt_lin.data.empty() ? nullptr : &dirt_lin);
    }

    Image out(W, H, 4);
    parallel_rows(H, [&](int y) {
        for (int x = 0; x < W; ++x) {
            const float* l = lin.px(x, y);
            float* o = out.px(x, y);
            float a = alpha[size_t(y) * W + x];
            float premult[3] = {l[0], l[1], l[2]};
            if (p.view != View::Source) {
                const float* g = glow.px(x, y);
                const float coverage = std::clamp(std::max({g[0], g[1], g[2]}), 0.0f, 1.0f);
                if (p.view == View::Glow) {
                    for (int c = 0; c < 3; ++c) premult[c] = g[c];
                    a = coverage;
                } else {
                    // Behind: the glow passes under the layer, through its transparency.
                    const float under = p.view == View::Behind ? 1.0f - a : 1.0f;
                    for (int c = 0; c < 3; ++c) {
                        const float screen = 1.0f - std::clamp(premult[c], 0.0f, 1.0f);
                        // Screen is a + b - ab with the glow capped at 1, so it never blows out.
                        premult[c] += under * (p.blend == Blend::Screen ? std::min(g[c], 1.0f) * screen : g[c]);
                    }
                    a = a + (1.0f - a) * coverage;  // glow extends the layer's coverage
                }
            }
            // Tonemap and encode straight colour, then premultiply again.
            for (int c = 0; c < 3; ++c) {
                float v = a > 0.0f ? premult[c] / a : 0.0f;
                if (p.tonemap) v = aces_fitted(v);
                if (p.gamma_correct) v = linear_to_srgb(v);
                o[c] = v * a;
            }
            o[3] = a;
        }
    });
    return out;
}

}  // namespace glow_profond
