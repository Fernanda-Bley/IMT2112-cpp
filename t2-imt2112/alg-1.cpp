#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <fstream> // Para guardar en archivo
#include <iostream> // Para std::cerr
#include <string>    // Para std::string

#define MATRIX_SIZE 2048
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
    srand(42); 
    
    // Configuración
    int threads_list[] = {1, 2, 4, 8};
    int num_configs = sizeof(threads_list) / sizeof(threads_list[0]);
    
    printf("=== Algoritmo 1: Paralelismo de Datos con OpenMP ===\n");
    printf("Tamaño de matriz: N = %d\n", MATRIX_SIZE);
    
    // Generar matrices
    float* A = matrix_generator(MATRIX_SIZE, RANGE);
    float* B = matrix_generator(MATRIX_SIZE, RANGE);
    float* C_ref = (float*) calloc(MATRIX_SIZE * MATRIX_SIZE, sizeof(float));
    
    printf("Ejecutando versión secuencial de referencia...\n");
    auto start_seq = std::chrono::high_resolution_clock::now();
    matmul_sequential(A, B, C_ref, MATRIX_SIZE);
    auto end_seq = std::chrono::high_resolution_clock::now();
    double time_seq = std::chrono::duration<double, std::milli>(end_seq - start_seq).count();
    
    // Abrir archivo para guardar resultados
    std::ofstream outfile("resultados-alg1.txt");
    if (!outfile.is_open()) {
        std::cerr << "Error al abrir el archivo de salida." << std::endl;
        return 1;
    }

    // Escribir encabezado en archivo y consola
    std::string header = "Algoritmo 1: OpenMP Data Parallelism\nN: " + std::to_string(MATRIX_SIZE) + "\n";
    std::string cols = "Threads\tTiempo (ms)\tSpeedup (vs Seq)\tEficiencia (vs Seq)\tSpeedup (vs P=1)\tEficiencia (vs P=1)\n";
    
    std::cout << header;
    std::cout << "Tiempo Secuencial Referencia: " << time_seq << " ms\n\n";
    std::cout << cols;
    
    outfile << header;
    outfile << "Tiempo Secuencial Referencia: " << time_seq << " ms\n";
    outfile << cols;

    double time_p1 = 0.0; // Para guardar el tiempo con 1 thread y calcular speedup relativo

    for (int t = 0; t < num_configs; t++) {
        int p = threads_list[t];
        float* C = (float*) calloc(MATRIX_SIZE * MATRIX_SIZE, sizeof(float));
        
        auto start = std::chrono::high_resolution_clock::now();
        matmul_parallel_omp(A, B, C, MATRIX_SIZE, p);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_p = std::chrono::duration<double, std::milli>(end - start).count();
        
        // Guardar tiempo de p=1 para cálculos posteriores
        if (p == 1) {
            time_p1 = time_p;
        }

        // Calcular métricas
        // Opción A: Comparado con secuencial puro
        double speedup_seq = time_seq / time_p;
        double efficiency_seq = speedup_seq / p;
        
        // Opción B: Comparado con paralelismo de 1 hilo (más justo para overhead de OpenMP)
        double speedup_p1 = (time_p1 > 0) ? time_p1 / time_p : 0;
        double efficiency_p1 = (p == 1) ? 1.0 : speedup_p1 / p;

        // Imprimir en consola
        printf("%d\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\n", 
               p, time_p, speedup_seq, efficiency_seq, speedup_p1, efficiency_p1);
        
        // Guardar en archivo
        outfile << p << "\t" << time_p << "\t" 
                << speedup_seq << "\t" << efficiency_seq << "\t"
                << speedup_p1 << "\t" << efficiency_p1 << "\n";
        
        free_matrix(C);
    }
    
    outfile.close();
    printf("\nResultados guardados en resultados_alg1_omp.txt\n");
    
    // Liberar memoria
    free_matrix(A);
    free_matrix(B);
    free_matrix(C_ref);
    
    return 0;
}