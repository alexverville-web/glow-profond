// glowprofond: apply the Glow Profond effect to an image file.
//
//   glowprofond in.png out.png --radius 200 --exposure 1 --tint 1,1,1 --tint 0.3,0.3,1
//
// Reads PNG/JPG/TGA/BMP/HDR (via stb) or PFM; writes PNG or PFM (float).
// 8-bit inputs are read as 0..1 values in the comp's working space.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "glow_profond/glow.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using glow_profond::GlowParams;
using glow_profond::Image;

namespace {

bool ends_with(const std::string& s, const char* suffix) {
    const size_t n = std::strlen(suffix);
    if (s.size() < n) return false;
    for (size_t i = 0; i < n; ++i) {
        if (std::tolower(s[s.size() - n + i]) != suffix[i]) return false;
    }
    return true;
}

bool read_pfm(const std::string& path, Image& img) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string magic;
    int w = 0, h = 0;
    float scale = 0.0f;
    f >> magic >> w >> h >> scale;
    f.get();
    if ((magic != "PF" && magic != "Pf") || w <= 0 || h <= 0 || scale >= 0.0f) return false;  // little-endian only
    const int c = magic == "PF" ? 3 : 1;
    std::vector<float> rows(size_t(w) * h * c);
    f.read(reinterpret_cast<char*>(rows.data()), std::streamsize(rows.size() * sizeof(float)));
    if (!f) return false;
    img = Image(w, h, 3);
    for (int y = 0; y < h; ++y) {
        const float* src = rows.data() + size_t(h - 1 - y) * w * c;  // PFM rows are bottom-up
        for (int x = 0; x < w; ++x) {
            for (int ch = 0; ch < 3; ++ch) img.px(x, y)[ch] = src[size_t(x) * c + (c == 3 ? ch : 0)];
        }
    }
    return true;
}

bool write_pfm(const std::string& path, const Image& img) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << "PF\n" << img.width << " " << img.height << "\n-1.0\n";
    std::vector<float> row(size_t(img.width) * 3);
    for (int y = img.height - 1; y >= 0; --y) {
        for (int x = 0; x < img.width; ++x) {
            for (int ch = 0; ch < 3; ++ch) row[size_t(x) * 3 + ch] = img.px(x, y)[ch];
        }
        f.write(reinterpret_cast<const char*>(row.data()), std::streamsize(row.size() * sizeof(float)));
    }
    return bool(f);
}

bool read_image(const std::string& path, Image& img) {
    if (ends_with(path, ".pfm")) return read_pfm(path, img);
    int w = 0, h = 0, n = 0;
    float* data = stbi_is_hdr(path.c_str()) ? stbi_loadf(path.c_str(), &w, &h, &n, 4) : nullptr;
    if (data) {
        img = Image(w, h, 4);
        std::memcpy(img.data.data(), data, img.data.size() * sizeof(float));
        stbi_image_free(data);
        return true;
    }
    unsigned char* bytes = stbi_load(path.c_str(), &w, &h, &n, 4);
    if (!bytes) return false;
    img = Image(w, h, 4);
    for (size_t i = 0; i < img.data.size(); ++i) img.data[i] = float(bytes[i]) / 255.0f;
    stbi_image_free(bytes);
    return true;
}

bool write_image(const std::string& path, const Image& img) {
    if (ends_with(path, ".pfm")) return write_pfm(path, img);
    std::vector<unsigned char> bytes(size_t(img.width) * img.height * 3);
    for (size_t i = 0; i < bytes.size(); ++i) {
        const float v = std::min(std::max(img.data[i], 0.0f), 1.0f);
        bytes[i] = static_cast<unsigned char>(v * 255.0f);  // truncation, like the reference demo
    }
    return stbi_write_png(path.c_str(), img.width, img.height, 3, bytes.data(), img.width * 3) != 0;
}

glow_profond::Rgb parse_rgb(const char* s) {
    glow_profond::Rgb c{1.0f, 1.0f, 1.0f};
    std::stringstream ss(s);
    std::string part;
    for (int i = 0; i < 3 && std::getline(ss, part, ','); ++i) c[i] = std::strtof(part.c_str(), nullptr);
    return c;
}

int usage() {
    std::cerr << "usage: glowprofond IN OUT [options]\n"
                 "  --radius F            largest blur sigma in px (400)\n"
                 "  --exposure F          stops (0)\n"
                 "  --threshold F         linear light (0)\n"
                 "  --threshold-smooth F  soft knee (0.25)\n"
                 "  --spread F            falloff exponent, 2 = 1/r^2 (2)\n"
                 "  --aspect F            >1 horizontal (1)\n"
                 "  --rgb-radius R,G,B    per-channel radius multipliers (1,1,1)\n"
                 "  --tint R,G,B          repeat for core -> halo gradient (1,1,1)\n"
                 "  --base-sigma F        tightest level sigma (1.5)\n"
                 "  --downsample N        extra global downsample, power of two (1)\n"
                 "  --view final|glow|source\n"
                 "  --unmult  --tonemap  --linear (disable gamma correction)\n";
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) return usage();
    const std::string in = argv[1], out = argv[2];
    GlowParams p;
    bool tint_set = false;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has_value = i + 1 < argc;
        auto next = [&]() { return argv[++i]; };
        if (a == "--radius" && has_value) p.radius = std::strtof(next(), nullptr);
        else if (a == "--exposure" && has_value) p.exposure = std::strtof(next(), nullptr);
        else if (a == "--threshold" && has_value) p.threshold = std::strtof(next(), nullptr);
        else if (a == "--threshold-smooth" && has_value) p.threshold_smooth = std::strtof(next(), nullptr);
        else if (a == "--spread" && has_value) p.spread = std::strtof(next(), nullptr);
        else if (a == "--aspect" && has_value) p.aspect = std::strtof(next(), nullptr);
        else if (a == "--rgb-radius" && has_value) p.rgb_radius = parse_rgb(next());
        else if (a == "--base-sigma" && has_value) p.base_sigma = std::strtof(next(), nullptr);
        else if (a == "--downsample" && has_value) p.downsample = std::atoi(next());
        else if (a == "--tint" && has_value) {
            if (!tint_set) p.tint.clear();
            tint_set = true;
            p.tint.push_back(parse_rgb(next()));
        } else if (a == "--view" && has_value) {
            const std::string v = next();
            p.view = v == "glow" ? glow_profond::View::Glow
                   : v == "source" ? glow_profond::View::Source
                                   : glow_profond::View::Final;
        } else if (a == "--unmult") p.unmult = true;
        else if (a == "--tonemap") p.tonemap = true;
        else if (a == "--linear") p.gamma_correct = false;
        else return usage();
    }

    Image src;
    if (!read_image(in, src)) {
        std::cerr << "cannot read " << in << "\n";
        return 1;
    }
    const Image result = glow_profond::render(src, p);
    if (!write_image(out, result)) {
        std::cerr << "cannot write " << out << "\n";
        return 1;
    }
    return 0;
}
