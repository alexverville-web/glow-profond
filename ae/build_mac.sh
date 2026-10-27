#!/bin/bash
# Build GlowProfond.plugin with only Apple's Command Line Tools (no Xcode.app,
# no CMake). Same steps as ae/CMakeLists.txt.
#
#   AE_SDK_ROOT=~/Documents/AE_Development/AfterEffectsSDK ae/build_mac.sh
#
# Output: build-mac/GlowProfond.plugin (universal arm64 + x86_64).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${AE_SDK_ROOT:?set AE_SDK_ROOT to the folder that contains Examples/Headers}"
OUT="${OUT_DIR:-$REPO/build-mac}"
VERSION="$(sed -n 's/^project(GlowProfond VERSION \([0-9.]*\).*/\1/p' "$REPO/CMakeLists.txt")"
BUNDLE="$OUT/GlowProfond.plugin"
SYSROOT="$(xcrun --show-sdk-path)"

[ -f "$SDK/Examples/Headers/AE_Effect.h" ] || { echo "AE_Effect.h not found under $SDK/Examples/Headers" >&2; exit 1; }

rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/Contents/MacOS" "$BUNDLE/Contents/Resources" "$OUT/obj"

echo "Compiling…"
xcrun clang++ -std=c++17 -O2 -arch arm64 -arch x86_64 -mmacosx-version-min=11.0 \
  -fvisibility=hidden -isysroot "$SYSROOT" \
  -I"$REPO/core/include" -I"$REPO/ae" \
  -I"$SDK/Examples/Headers" -I"$SDK/Examples/Headers/SP" \
  -I"$SDK/Examples/Util" -I"$SDK/Examples/Resources" \
  -bundle -o "$BUNDLE/Contents/MacOS/GlowProfond" \
  "$REPO/core/src/glow.cpp" "$REPO/ae/GlowProfond.cpp"

echo "Compiling PiPL…"
xcrun Rez -o "$BUNDLE/Contents/Resources/GlowProfond.rsrc" -useDF -script Roman \
  -d __MACH__=1 -arch x86_64 -arch arm64 \
  -i "$REPO/ae" -i "$SDK/Examples/Headers" -i "$SDK/Examples/Headers/SP" \
  -i "$SDK/Examples/Resources" -isysroot "$SYSROOT" \
  "$REPO/ae/GlowProfondPiPL.r"

sed -e 's/${MACOSX_BUNDLE_EXECUTABLE_NAME}/GlowProfond/' \
    -e "s/\${MACOSX_BUNDLE_SHORT_VERSION_STRING}/$VERSION/" \
    "$REPO/ae/Info.plist.in" > "$BUNDLE/Contents/Info.plist"
printf 'eFKTFXTC' > "$BUNDLE/Contents/PkgInfo"

codesign --force --sign - "$BUNDLE"
echo "Built $BUNDLE"
