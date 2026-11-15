#!/usr/bin/env bash
# Mirror predicted RTTMs from benchmark/outputs/rttm/<system>/<dataset>/ into
# Dataset/<Dataset>/Predicted RTTM/<System>/ (copies only; safe to re-run).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
declare -A DS=( [msdwild]="MSDWild" [voxconverse]="VoxConverse" )
declare -A SYS=( [nemotron]="Nemotron" [vbx]="VBx" )
for s in "${!SYS[@]}"; do for d in "${!DS[@]}"; do
  src="$ROOT/benchmark/outputs/rttm/$s/$d"; [ -d "$src" ] || continue
  dst="$ROOT/Dataset/${DS[$d]}/Predicted RTTM/${SYS[$s]}"; mkdir -p "$dst"
  cp -u "$src"/*.rttm "$dst"/ 2>/dev/null || true
  echo "${SYS[$s]} ${DS[$d]}: $(ls "$dst" | wc -l) files"
done; done
