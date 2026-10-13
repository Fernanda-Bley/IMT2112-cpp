#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <chrono>
#include <cmath>

#define DEFAULT_N 2048
#define DEFAULT_BLOCK 64
#define RANGE 10

// Función que genera una matriz cuadrada almacenada en formato 1D (row-major)
float* matrix_generator(int n, int range)
{
    float* matrix = (float*) calloc(n * n, sizeof(float));
    
    for (int i = 0; i < n * n; i++)
    {
        matrix[i] = (2 * range * ((float)rand() / (float)RAND_MAX)) - range;
    }
    
    return matrix;
}

// Función para liberar memoria
void free_matrix(float* matrix)
{
    free(matrix);
}

// Multiplicación densa base (usada en las hojas del árbol)
void matmul_base(float* A, float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// Función auxiliar para sumar dos submatrices
void add_blocks(float* C, float* A, float* B, int n, 
                int row_offset, int col_offset, int stride) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[(row_offset + i) * stride + (col_offset + j)] += 
                A[i * n + j] + B[i * n + j];
        }
    }
}

// Función auxiliar para multiplicar y acumular en un bloque específico de C
void multiply_and_accumulate(float* C, float* A, float* B, int n,
                             int row_off_a, int col_off_a, int stride_a,
                             int row_off_b, int col_off_b, int stride_b,
                             int row_off_c, int col_off_c, int stride_c) {
    
    // Extraer subbloques temporalmente o calcular índices directamente
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            float a_val = A[(row_off_a + i) * stride_a + (col_off_a + k)];
            for (int j = 0; j < n; ++j) {
                float b_val = B[(row_off_b + k) * stride_b + (col_off_b + j)];
                C[(row_off_c + i) * stride_c + (col_off_c + j)] += a_val * b_val;
            }
        }
    }
}

// Algoritmo 2: Dividir y Conquistar Paralelo
void matmul_divide_conquer_parallel(float* A, float* B, float* C, int N, int block_size) {
    
    // Caso base: si el tamaño es menor o igual al bloque base, usar multiplicación densa
    if (N <= block_size) {
        matmul_base(A, B, C, N);
        return;
    }
    
    int half = N / 2;
    
    // Paralelismo de tareas: los 8 productos de bloques son independientes
    #pragma omp parallel sections
    {
        // C11 = A11*B11 + A12*B21
        #pragma omp section
        {
            // Inicializar bloque C11 a cero
            for (int i = 0; i < half; ++i)
                for (int j = 0; j < half; ++j)
                    C[i * N + j] = 0.0f;
            
            // A11 * B11 -> C11
            matmul_divide_conquer_parallel(A, B, C, half, block_size);
            
            // A12 * B21 -> C11 (acumular)
            multiply_and_accumulate(C, A, B, half,
                                   0, half, N,      // A12
                                   half, 0, N,      // B21
                                   0, 0, N);        // C11
        }
        
        // C12 = A11*B12 + A12*B22
        #pragma omp section
        {
            for (int i = 0; i < half; ++i)
                for (int j = 0; j < half; ++j)
                    C[i * N + (half + j)] = 0.0f;
            
            matmul_divide_conquer_parallel(A, B + half, C + half, half, block_size);
            multiply_and_accumulate(C, A, B, half,
                                   0, half, N,
                                   half, half, N,
                                   0, half, N);
        }
        
        // C21 = A21*B11 + A22*B21
        #pragma omp section
        {
            for (int i = 0; i < half; ++i)
                for (int j = 0; j < half; ++j)
                    C[(half + i) * N + j] = 0.0f;
            
            matmul_divide_conquer_parallel(A + half * N, B, C + half * N, half, block_size);
            multiply_and_accumulate(C, A, B, half,
                                   half, 0, N,
                                   half, 0, N,
                                   half, 0, N);
        }
        
        // C22 = A21*B12 + A22*B22
        #pragma omp section
        {
            for (int i = 0; i < half; ++i)
                for (int j = 0; j < half; ++j)
                    C[(half + i) * N + (half + j)] = 0.0f;
            
            matmul_divide_conquer_parallel(A + half * N, B + half, C + half * N + half, half, block_size);
            multiply_and_accumulate(C, A, B, half,
                                   half, half, N,
                                   half, half, N,
                                   half, half, N);
        }
    }
}

int main(int argc, char* argv[]) {
    int N = DEFAULT_N;
    int block_size = DEFAULT_BLOCK;

    // Leer tamaño del bloque
    if (argc > 1) {
        block_size = atoi(argv[1]); // Convierte string a entero
    }
    
    // Validación básica
    if (block_size <= 0 || block_size > N) {
        printf("Error: El tamaño de bloque debe estar entre 1 y %d\n", N);
        return 1;
    }

    srand(42);

    printf("=== Algoritmo 2: Dividir y Conquistar ===\n");
    printf("N = %d, Block Size (n) = %d\n", N, block_size);
    printf("Profundidad del árbol: %.2f\n", log2((double)N / block_size));
    
    // Generar matrices
    float* A = matrix_generator(N, RANGE);
    float* B = matrix_generator(N, RANGE);
    float* C = (float*)calloc(N * N, sizeof(float));

    // Warm-up
    matmul_divide_conquer_parallel(A, B, C, N, block_size); 
    // Reiniciar C a 0 después del warm-up si lo usas
    for(int i=0; i<N*N; i++) C[i] = 0.0f;

    // Medición
    auto start = std::chrono::high_resolution_clock::now();
    
    matmul_divide_conquer_parallel(A, B, C, N, block_size);
    
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    printf("Tiempo de ejecución: %.4f ms\n", time_ms);

    // Liberar memoria
    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

    return 0;
}