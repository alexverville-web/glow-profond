#!/bin/bash
# Installs Glow Profond into Adobe's shared plug-in folder (every After
# Effects version sees it). Double-click it, or: ./"Install Glow Profond.command" --dry-run
set -euo pipefail
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/GlowProfond.plugin"
DST="/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore"

echo "Glow Profond — installation"
if [ ! -f "$SRC/Contents/MacOS/GlowProfond" ]; then
  echo "ERROR: GlowProfond.plugin not found next to this installer ($SRC)."
  exit 1
fi
if pgrep -xq "After Effects"; then
  echo "Please quit After Effects first, then run this installer again."
  echo "Quittez d'abord After Effects, puis relancez cet installeur."
  exit 1
fi
if [ ! -d "$DST" ]; then
  echo "ERROR: $DST does not exist. Is After Effects installed?"
  exit 1
fi

# sudo only when the folder isn't writable for this account.
SUDO=""
[ -w "$DST" ] || SUDO="sudo"
run() { if [ $DRY = 1 ]; then echo "[dry-run] $*"; else "$@"; fi; }

if [ -e "$DST/GlowProfond.plugin" ]; then
  echo "Replacing the installed Glow Profond…"
  # Move aside first (works even for a root-owned leftover), then delete.
  OLD="$DST/.GlowProfond-old-$$"
  run $SUDO mv "$DST/GlowProfond.plugin" "$OLD"
  run $SUDO rm -rf "$OLD" || true
fi
run $SUDO cp -R "$SRC" "$DST/"
run $SUDO xattr -dr com.apple.quarantine "$DST/GlowProfond.plugin" 2>/dev/null || true

if [ $DRY = 1 ]; then
  echo "Dry run: nothing was changed."
  exit 0
fi
if codesign -v "$DST/GlowProfond.plugin" 2>/dev/null; then
  echo ""
  echo "Glow Profond is installed. Start After Effects: Effect > Gimmick > Glow Profond."
  echo "Glow Profond est installé. Lancez After Effects : Effet > Gimmick > Glow Profond."
else
  echo "WARNING: the installed plug-in failed its signature check."
  exit 1
fi
