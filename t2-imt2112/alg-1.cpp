//  para correrlo hay que compilar g++ -fopenmp -O2 -o algoritmo1 alg-1.cpp
// Se llama con ./algoritmo1 N (Tamaño de matriz)

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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



int main(int argc, char* argv[])
{
    srand(42); 
    
    // 1. Lectura de argumentos igual que en alg2.cpp
    int N = DEFAULT_N;
    int block_size = DEFAULT_BLOCK; // Aunque en Alg1 base no se usa mucho, lo mantenemos por consistencia
    
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2) {
        block_size = atoi(argv[2]);
    }

    // Configuración de threads
    std::vector<int> threads_list = {1, 2, 4, 8};
    
    printf("=== Algoritmo 1: Paralelismo de Datos con OpenMP ===\n");
    printf("N = %d\n", N);
    
    // Generar matrices basadas en el N leído
    float* A = matrix_generator(N, RANGE);
    float* B = matrix_generator(N, RANGE);
    float* C_ref = (float*) calloc(N * N, sizeof(float));
    
    printf("Ejecutando versión secuencial de referencia...\n");
    auto start_seq = std::chrono::high_resolution_clock::now();
    matmul_sequential(A, B, C_ref, N);
    auto end_seq = std::chrono::high_resolution_clock::now();
    double time_seq = std::chrono::duration<double, std::milli>(end_seq - start_seq).count();
    
    // Nombre del archivo dinámico o fijo según prefieras
    std::string filename = "resultados-alg1.txt";
    std::ofstream outfile(filename);
    
    if (!outfile.is_open()) {
        std::cerr << "Error al abrir el archivo de salida." << std::endl;
        return 1;
    }

    // Encabezados
    std::string header = "Algoritmo 1: OpenMP Data Parallelism\nN: " + std::to_string(N) + "\n";
    std::string cols = "Threads\tTiempo (ms)\tSpeedup (vs Seq)\tEficiencia (vs Seq)\tSpeedup (vs P=1)\tEficiencia (vs P=1)\n";
    
    std::cout << header;
    std::cout << "Tiempo Secuencial Referencia: " << time_seq << " ms\n\n";
    std::cout << cols;
    
    outfile << header;
    outfile << "Tiempo Secuencial Referencia: " << time_seq << " ms\n";
    outfile << cols;

    double time_p1 = 0.0; 

    for (int p : threads_list) {
        float* C = (float*) calloc(N * N, sizeof(float));
        
        auto start = std::chrono::high_resolution_clock::now();
        matmul_parallel_omp(A, B, C, N, p);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_p = std::chrono::duration<double, std::milli>(end - start).count();
        
        if (p == 1) {
            time_p1 = time_p;
        }

        // Métricas
        double speedup_seq = time_seq / time_p;
        double efficiency_seq = speedup_seq / p;
        
        double speedup_p1 = (time_p1 > 0) ? time_p1 / time_p : 0;
        double efficiency_p1 = (p == 1) ? 1.0 : speedup_p1 / p;

        printf("%d\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\n", 
               p, time_p, speedup_seq, efficiency_seq, speedup_p1, efficiency_p1);
        
        outfile << p << "\t" << time_p << "\t" 
                << speedup_seq << "\t" << efficiency_seq << "\t"
                << speedup_p1 << "\t" << efficiency_p1 << "\n";
        
        free_matrix(C);
    }
    
    outfile.close();
    printf("\nResultados guardados en %s\n", filename.c_str());
    
    free_matrix(A);
    free_matrix(B);
    free_matrix(C_ref);
    
    return 0;
}