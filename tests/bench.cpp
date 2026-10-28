// Times render_premultiplied on a synthetic HD / UHD frame.
//   bench [width height radius repeats]
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "glow_profond/glow.hpp"

int main(int argc, char** argv) {
    const int w = argc > 1 ? std::atoi(argv[1]) : 1920;
    const int h = argc > 2 ? std::atoi(argv[2]) : 1080;
    glow_profond::GlowParams p;
    p.radius = argc > 3 ? float(std::atof(argv[3])) : 400.0f;
    const int reps = argc > 4 ? std::atoi(argv[4]) : 3;

    glow_profond::Image src(w, h, 4);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float* o = src.px(x, y);
            const bool lit = ((x / 64) + (y / 64)) % 7 == 0;
            o[0] = lit ? 1.0f : 0.05f;
            o[1] = lit ? 0.6f : 0.05f;
            o[2] = lit ? 0.2f : 0.05f;
            o[3] = 1.0f;
        }
    }
    double best = 1e9;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const glow_profond::Image out = glow_profond::render_premultiplied(src, p);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (ms < best) best = ms;
        if (out.data.empty()) return 1;
    }
    std::printf("%dx%d radius %.0f: %.1f ms\n", w, h, p.radius, best);
    return 0;
}
