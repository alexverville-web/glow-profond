#!/bin/bash
# Builds Glow Profond and packages a Mac installer zip:
#   dist/GlowProfond-<version>-mac.zip  containing  "Glow Profond <version> - Mac/"
#     GlowProfond.plugin, Install / Uninstall Glow Profond.command, README.txt
#
#   AE_SDK_ROOT=~/Documents/AE_Development/AfterEffectsSDK ae/package_mac.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="$(sed -n 's/^project(GlowProfond VERSION \([0-9.]*\).*/\1/p' "$REPO/CMakeLists.txt")"
NAME="Glow Profond $VERSION - Mac"
DIST="$REPO/dist"
STAGE="$DIST/stage"
ZIP="$DIST/GlowProfond-$VERSION-mac.zip"

"$REPO/ae/build_mac.sh"

rm -rf "$STAGE" "$ZIP"
mkdir -p "$STAGE/$NAME"
cp -R "$REPO/build-mac/GlowProfond.plugin" "$STAGE/$NAME/"
cp "$REPO/ae/package/Install Glow Profond.command" "$REPO/ae/package/Uninstall Glow Profond.command" "$STAGE/$NAME/"
sed "s/__VERSION__/$VERSION/" "$REPO/ae/package/README.txt" > "$STAGE/$NAME/README.txt"
chmod +x "$STAGE/$NAME/"*.command
codesign -v "$STAGE/$NAME/GlowProfond.plugin"

# ditto keeps the bundle, its signature and the exec bits.
ditto -c -k --sequesterRsrc --keepParent "$STAGE/$NAME" "$ZIP"
rm -rf "$STAGE"
echo "Packaged $ZIP"
