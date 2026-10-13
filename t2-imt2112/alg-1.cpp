#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <chrono>
#include <cmath>

#define MATRIX_SIZE 2048  // Cambiado de N a MATRIX_SIZE para evitar conflicto
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

// Algoritmo 1: Multiplicación secuencial (para referencia)
void matmul_sequential(float* A, float* B, float* C, int n)
{
    // Inicializar C a cero
    for (int i = 0; i < n * n; i++) {
        C[i] = 0.0f;
    }
    
    for(int i = 0; i < n; ++i) {
        for(int k = 0; k < n; ++k) {
            for(int j = 0; j < n; ++j) {
                C[i*n + j] += A[i*n + k] * B[k*n + j];
            }
        }
    }
}

// Algoritmo 1: Multiplicación paralela con OpenMP
void matmul_parallel_omp(float* A, float* B, float* C, int n, int num_threads)
{
    // Inicializar C a cero
    for (int i = 0; i < n * n; i++) {
        C[i] = 0.0f;
    }
    
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < n; ++i) {
        for(int k = 0; k < n; ++k) {
            for(int j = 0; j < n; ++j) {
                C[i*n + j] += A[i*n + k] * B[k*n + j];
            }
        }
    }
}


int main()
{
    srand(42); // Semilla fija para reproducibilidad
    
    printf("=== Algoritmo 1: Paralelismo de Datos con OpenMP ===\n");
    printf("Tamaño de matriz: N = %d\n", MATRIX_SIZE);
    printf("Número total de elementos: %d\n", MATRIX_SIZE * MATRIX_SIZE);
    printf("\n");
    
    // Generar matrices
    float* A = matrix_generator(MATRIX_SIZE, RANGE);
    float* B = matrix_generator(MATRIX_SIZE, RANGE);
    float* C_ref = (float*) calloc(MATRIX_SIZE * MATRIX_SIZE, sizeof(float));
    
    printf("Matrices generadas.\n\n");
    
    // Ejecutar versión secuencial para obtener tiempo de referencia
    printf("Ejecutando versión secuencial...\n");
    auto start_seq = std::chrono::high_resolution_clock::now();
    matmul_sequential(A, B, C_ref, MATRIX_SIZE);
    auto end_seq = std::chrono::high_resolution_clock::now();
    double time_seq = std::chrono::duration<double, std::milli>(end_seq - start_seq).count();
    printf("Tiempo secuencial: %.2f ms\n\n", time_seq);
    
    // Probar con diferentes números de threads
    int threads[] = {1, 2, 4, 8};
    int num_configs = sizeof(threads) / sizeof(threads[0]);
    
    printf("%-10s %-15s %-15s %-15s\n", "Threads", "Tiempo (ms)", "Speedup", "Eficiencia");
    printf("%-10s %-15s %-15s %-15s\n", "-------", "-----------", "-------", "----------");
    
    double times[num_configs];
    
    for (int t = 0; t < num_configs; t++) {
        int p = threads[t];
        float* C = (float*) calloc(MATRIX_SIZE * MATRIX_SIZE, sizeof(float));
        
        auto start = std::chrono::high_resolution_clock::now();
        matmul_parallel_omp(A, B, C, MATRIX_SIZE, p);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_p = std::chrono::duration<double, std::milli>(end - start).count();
        times[t] = time_p;
        
        // Calcular speedup y eficiencia
        double speedup = time_seq / time_p;
        double efficiency = speedup / p;
        
        printf("%-10d %-15.2f %-15.2f %-15.2f\n", p, time_p, speedup, efficiency);
        
        // Verificación opcional (comentar para producción)
        if (t == 0) {
            bool correct = true;
            for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE && i < 1000; i++) {
                if (std::abs(C[i] - C_ref[i]) > 0.1) {
                    correct = false;
                    break;
                }
            }
            if (!correct) {
                printf("¡ADVERTENCIA! Resultados incorrectos para %d threads\n", p);
            }
        }
        
        free_matrix(C);
    }
    
    printf("\n");
    
    // Liberar memoria
    free_matrix(A);
    free_matrix(B);
    free_matrix(C_ref);
    
    return 0;
}