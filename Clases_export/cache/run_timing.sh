#!/usr/bin/env bash
# Uso: ./scripts/run_timing.sh <directorio_bin> [<n1> <n2> ...]
#
# TODO: ajustar los tamaños por defecto según la máquina del presentador.
#   Ejecutar `lscpu` y elegir tamaños que:
#     - quepan en L1 (~32 KB  → n ≈ 64  para matrices double)
#     - quepan en L2/L3       → n ≈ 256–512
#     - excedan la LLC        → n ≈ 2048–4096
#
# Los tamaños por defecto son una guía razonable para una máquina típica.

BINDIR="${1:?Uso: $0 <directorio_bin> [n1 n2 ...]}"
shift

SIZES=(${@:-64 256 1024 2048})

RESULTS_DIR="results"
CSV="$RESULTS_DIR/timing.csv"

mkdir -p "$RESULTS_DIR"

if [[ ! -f "$CSV" ]]; then
    echo "kernel,n,bindir,time_s,checksum" > "$CSV"
fi

for KERNEL in "$BINDIR"/*; do
    [[ -x "$KERNEL" ]] || continue
    NAME="$(basename "$KERNEL")"
    for N in "${SIZES[@]}"; do
        echo -n "  $NAME  n=$N ... "
        OUTPUT="$("$KERNEL" "$N")"
        TIME_S="$(echo "$OUTPUT" | grep -oP 'time_s=\K[0-9.]+')"
        CHECKSUM="$(echo "$OUTPUT" | grep -oP 'checksum=\K[0-9.]+')"
        echo "$NAME,$N,$BINDIR,$TIME_S,$CHECKSUM" >> "$CSV"
        echo "$TIME_S s"
    done
done

echo "Resultados acumulados en $CSV"
