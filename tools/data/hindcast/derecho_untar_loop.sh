#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-License-Identifier: Apache-2.0
#
# Derecho-side consumer (LIGHT, login-node ok): watch the tar landing dir, untar
# each per-year zarr store (plain tar -> pure I/O, no decompress) into the final
# hindcasts_dsi/zarr/<model>/ layout, drop a .ready marker (consumed by the
# develop-queue h5-expansion job), and delete the tar. Runs until an ALLDONE
# sentinel appears and no tars remain.
#   nohup bash derecho_untar_loop.sh >/glade/derecho/scratch/awikner/hindcast_tars/untar.log 2>&1 &

set -uo pipefail
DTAR=/glade/derecho/scratch/awikner/hindcast_tars
OUT=/glade/derecho/scratch/awikner/hindcasts_dsi
READY="$OUT/zarr/.ready"
mkdir -p "$DTAR" "$OUT/zarr" "$OUT/h5" "$READY"
echo "======== [$(date)] untar loop start ========"

model_of() { local b="$1"; b="${b%.tar}"; b="${b%_zarr}"; echo "${b%_*}"; }   # models may contain underscores
year_of()  { local b="$1"; b="${b%.tar}"; b="${b%_zarr}"; echo "${b##*_}"; }

while true; do
  shopt -s nullglob
  progressed=0
  for t in "$DTAR"/*_zarr.tar; do
    # Only extract a tar that has FULLY arrived: require its size to be stable
    # (Globus writes the dest file in place, so a mid-transfer tar is still
    # growing -> extracting it yields "Unexpected EOF").
    s1=$(stat -c %s "$t" 2>/dev/null || echo 0); sleep 8
    s2=$(stat -c %s "$t" 2>/dev/null || echo 0)
    [ "$s1" = "$s2" ] && [ "$s1" -gt 0 ] || { echo "[$(date)] $(basename "$t") still arriving ($s1->$s2), skip"; continue; }
    b=$(basename "$t"); m=$(model_of "$b"); y=$(year_of "$b"); mkdir -p "$OUT/zarr/$m"
    # verify archive integrity before committing (guards against a paused transfer)
    if tar -tf "$t" >/dev/null 2>&1 && tar -xf "$t" -C "$OUT/zarr/$m"; then
      rm -f "$t"; touch "$READY/${m}_${y}.ready"; echo "[$(date)] untarred $b"; progressed=1
    else echo "[$(date)] [retry] $b not complete/valid yet"; fi
  done
  if [ -f "$DTAR/ALLDONE" ] && [ -z "$(ls "$DTAR"/*_zarr.tar 2>/dev/null)" ]; then
    echo "[$(date)] ALLDONE + drained"; break
  fi
  [ "$progressed" -eq 0 ] && sleep 60
done
echo "======== [$(date)] untar loop finished ========"
