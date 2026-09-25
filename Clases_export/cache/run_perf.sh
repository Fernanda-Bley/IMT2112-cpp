#!/usr/bin/env bash
# Uso: ./scripts/run_perf.sh <binario> <n>
#

BINARY="${1:?Uso: $0 <binario> <n>}"
N="${2:?Uso: $0 <binario> <n>}"

perf stat \
    -e instructions,cycles,cache-references,cache-misses \
    "$BINARY" "$N"
