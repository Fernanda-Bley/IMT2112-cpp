#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream> // Para guardar en archivo si lo deseas

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
    
    if (N <= block_size) {
        matmul_base(A, B, C, N);
        return;
    }
    
    int half = N / 2;
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for (int i = 0; i < half; ++i)
                for (int j = 0; j < half; ++j)
                    C[i * N + j] = 0.0f;
            
            matmul_divide_conquer_parallel(A, B, C, half, block_size);
            multiply_and_accumulate(C, A, B, half,
                                   0, half, N,      
                                   half, 0, N,      
                                   0, 0, N);        
        }
        
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
    
    // Lista de threads a probar
    std::vector<int> threads_list = {1, 2, 4, 8};

    // Leer tamaño del bloque desde argumentos si existe
    if (argc > 1) {
        N = atoi(argv[1]); 
    }
    
    if (block_size <= 0 || block_size > N) {
        printf("Error: El tamaño de bloque debe estar entre 1 y %d\n", N);
        return 1;
    }

    srand(42);

    printf("=== Algoritmo 2: Dividir y Conquistar ===\n");
    printf("N = %d, Block Size (n) = %d\n", N, block_size);
    
    // Generar matrices A y B una sola vez
    float* A = matrix_generator(N, RANGE);
    float* B = matrix_generator(N, RANGE);
    
    // Abrir archivo para guardar resultados
    std::ofstream outfile("resultados-alg-2.txt");
    if (!outfile.is_open()) {
        std::cerr << "Error al abrir el archivo de salida." << std::endl;
        return 1;
    }
    
    // Encabezado para el archivo
    outfile << "Algoritmo 2: Dividir y Conquistar\n";
    outfile << "N = " << N << ", Block Size = " << block_size << "\n";
    outfile << "Threads\tTiempo (ms)\tSpeedup (vs P=1)\tEficiencia (vs P=1)\n";
    outfile << "--------------------------------------------------------\n";

    // Encabezado para consola
    printf("%-10s %-15s %-15s %-15s\n", "Threads", "Tiempo (ms)", "Speedup", "Eficiencia");
    printf("%-10s %-15s %-15s %-15s\n", "-------", "-----------", "-------", "----------");

    double time_p1 = 0.0; // Variable para guardar el tiempo con 1 thread

    for (int p : threads_list) {
        // Crear una copia fresca de C para cada medición
        float* C = (float*)calloc(N * N, sizeof(float));
        
        // Establecer el número de threads para OpenMP
        omp_set_num_threads(p);
        
        // Warm-up
        matmul_divide_conquer_parallel(A, B, C, N, block_size);
        
        // Reiniciar C a 0 después del warm-up
        for(int i=0; i<N*N; i++) C[i] = 0.0f;

        // Medición real
        auto start = std::chrono::high_resolution_clock::now();
        
        matmul_divide_conquer_parallel(A, B, C, N, block_size);
        
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Guardar tiempo de p=1 para usarlo como referencia
        if (p == 1) {
            time_p1 = time_ms;
        }

        // Calcular Speedup y Eficiencia respecto a P=1
        double speedup = (time_p1 > 0) ? time_p1 / time_ms : 0.0;
        double efficiency = (p > 0) ? speedup / p : 0.0;

        // Imprimir en consola con formato tabulado
        printf("%-10d %-15.2f %-15.2f %-15.2f\n", p, time_ms, speedup, efficiency);
        
        // Guardar en archivo
        outfile << p << "\t" << time_ms << "\t" << speedup << "\t" << efficiency << "\n";

        // Liberar C actual antes de la siguiente iteración
        free_matrix(C);
    }
    
    outfile.close();
    printf("\nResultados guardados en resultados-alg-2.txt\n");

    // Liberar memoria de A y B
    free_matrix(A);
    free_matrix(B);

    return 0;
}