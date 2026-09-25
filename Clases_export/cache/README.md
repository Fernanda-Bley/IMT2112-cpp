# cache-lecture — demo de efectos de caché

Proyecto de demostración para la clase de memoria caché. Incluye implementaciones
de producto punto, producto matriz-vector y producto matriz-matriz, con variantes
que difieren en orden de loops.

## Binarios

| Binario         | Descripción                          |
|-----------------|--------------------------------------|
| `dot`           | Producto punto                       |
| `matvec_row`    | y = Ax, loop externo fila            |
| `matvec_col`    | y = Ax, loop externo columna         |
| `matmul_ijk`    | C = AB, orden de loops ijk           |
| `matmul_ikj`    | C = AB, orden de loops ikj           |
| `matmul_jki`    | C = AB, orden de loops jki           |

## Compilar

```bash
# Flags por defecto (-O2)
make

# Un directorio por combinación de flags
make BINDIR=bin/O0        CXXFLAGS="-O0 -std=c++14"
make BINDIR=bin/O2        CXXFLAGS="-O2 -std=c++14"
make BINDIR=bin/O3native  CXXFLAGS="-O3 -march=native -std=c++14"
make BINDIR=bin/O3fast    CXXFLAGS="-O3 -march=native -ffast-math -std=c++14"

make clean   # elimina todo el directorio bin/
```

## Ejecutar

```bash
./bin/matmul_ikj 1024        # n=1024, semilla por defecto
./bin/dot 512 7              # n=512, semilla=7
```

Salida: `<kernel> n=<n> time_s=<segundos> checksum=<valor>`

## Scripts de medición

```bash
# Tiempos para todos los kernels en bin/O2, tamaños 64 256 1024 2048
./scripts/run_timing.sh bin/O2 64 256 1024 2048

# perf stat para un binario específico
./scripts/run_perf.sh bin/O2/matmul_ijk 1024
```

Los resultados se acumulan en `results/timing.csv` y `results/perf.csv`.
Ejecutar con distintos `BINDIR` permite comparar flags en el mismo CSV.
