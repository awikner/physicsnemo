#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 The University of Chicago.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Fast zarr replication via tar-bundling.
#
# A converted per-year zarr store is ~17k tiny (~1.6 MB) chunk files — the
# per-timestep chunking is deliberate (training samples random timesteps), but
# Globus then pays ~20 ms of per-file overhead on millions of files, capping
# throughput far below link rate (~70 MB/s vs 500 MB/s-1 GB/s). Bundle each store
# into ONE tar for the wire (chunks are already blosc-compressed, so no -z),
# transfer the tars, untar at the destination. The on-disk chunking is preserved
# end to end; only the transport changes.
#
# Run `pack` + `send` on the SOURCE cluster, `unpack` on the DEST cluster:
#   ZARR_ROOT=/scratch/.../physicsnemo-zarr        replicate_tar.sh pack   <dataset>
#   SRC_UUID=<src-collection> GLOBUS=~/gcli/bin/globus \
#       replicate_tar.sh send <dataset> <DST_UUID> <DST_TAR_STAGE>
#   ZARR_ROOT=/glade/.../physicsnemo-zarr          replicate_tar.sh unpack <dataset>
#
# Env: ZARR_ROOT (zarr store root, required for pack/unpack); TAR_STAGE (tar
#   staging dir, default $ZARR_ROOT/../zarr-tars); GLOBUS (default
#   ~/gcli/bin/globus); NP (parallelism, default nproc); SRC_UUID (source
#   collection, required for send).

set -euo pipefail

MODE=${1:?usage: replicate_tar.sh pack|send|unpack <dataset> [...]}
DS=${2:?dataset name}
TAR_STAGE=${TAR_STAGE:-${ZARR_ROOT:-}/../zarr-tars}
GLOBUS=${GLOBUS:-$HOME/gcli/bin/globus}
NP=${NP:-$(nproc)}
stage="$TAR_STAGE/$DS"

case "$MODE" in
  pack)
    : "${ZARR_ROOT:?set ZARR_ROOT to the physicsnemo-zarr root}"
    src="$ZARR_ROOT/$DS"; mkdir -p "$stage"
    n=$(ls -d "$src"/*.zarr 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] || { echo "no $DS/*.zarr under $src" >&2; exit 1; }
    echo "[$(date)] pack: $n $DS stores -> $stage  (NP=$NP, uncompressed)"
    # One tar per store, in parallel; skip a tar that already exists (resumable).
    ls -d "$src"/*.zarr 2>/dev/null | sed "s|.*/||" | \
      xargs -P "$NP" -I{} sh -c 'test -s "$1/{}.tar" || tar -cf "$1/{}.tar" -C "$2" "{}"' _ "$stage" "$src"
    echo "[$(date)] pack done: $(ls "$stage"/*.tar 2>/dev/null | wc -l) tars, $(du -sh "$stage" 2>/dev/null | cut -f1)"
    ;;

  send)
    DST_UUID=${3:?dest Globus collection UUID}; DST_STAGE=${4:?dest tar-stage path}
    : "${SRC_UUID:?set SRC_UUID to the source Globus collection}"
    n=$(ls "$stage"/*.tar 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] || { echo "no tars in $stage — run pack first" >&2; exit 1; }
    echo "[$(date)] send: $n tars  $SRC_UUID:$stage -> $DST_UUID:$DST_STAGE/$DS"
    # Few big files → mtime sync is enough; Globus still integrity-checks each transfer.
    tid=$( ( cd "$stage" && for t in *.tar; do echo "$t $t"; done ) | \
      "$GLOBUS" transfer "$SRC_UUID:$stage" "$DST_UUID:$DST_STAGE/$DS" --batch - \
        --sync-level mtime --label "ai-rossby $DS tars" --format unix --jmespath task_id )
    echo "[$(date)] transfer task: $tid"
    echo "$tid"
    ;;

  unpack)
    : "${ZARR_ROOT:?set ZARR_ROOT to the physicsnemo-zarr root}"
    dst="$ZARR_ROOT/$DS"; mkdir -p "$dst"
    n=$(ls "$stage"/*.tar 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] || { echo "no tars in $stage" >&2; exit 1; }
    echo "[$(date)] unpack: $n $DS tars  $stage -> $dst  (NP=$NP)"
    ls "$stage"/*.tar 2>/dev/null | xargs -P "$NP" -I{} tar -xf {} -C "$dst"
    echo "[$(date)] unpack done: $dst has $(ls -d "$dst"/*.zarr 2>/dev/null | wc -l) stores"
    ;;

  *) echo "unknown mode: $MODE (want pack|send|unpack)" >&2; exit 2;;
esac
