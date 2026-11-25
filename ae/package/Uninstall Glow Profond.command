#!/bin/bash
# Removes Glow Profond from Adobe's shared plug-in folder.
set -euo pipefail
DST="/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore"
if pgrep -xq "After Effects"; then
  echo "Please quit After Effects first. / Quittez d'abord After Effects."
  exit 1
fi
if [ ! -e "$DST/GlowProfond.plugin" ]; then
  echo "Glow Profond is not installed. / Glow Profond n'est pas installé."
  exit 0
fi
SUDO=""
[ -w "$DST" ] || SUDO="sudo"
OLD="$DST/.GlowProfond-old-$$"
$SUDO mv "$DST/GlowProfond.plugin" "$OLD"
$SUDO rm -rf "$OLD" || true
echo "Glow Profond was removed. / Glow Profond a été désinstallé."
