#include <omp.h>
#include <iostream>
#include <vector>

int main(){
    int n = 1000;
    std::vector<double> a(n), b(n), c(n);

    // Inicializar a y b
    for(int i = 0; i < n; i++){
        a[i] = i;
        b[i] = 2.0 * i;
    }

    // Suma paralela: cada hilo calcula un subconjunto de iteraciones
    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        c[i] = a[i] + b[i];
    }

    // Imprimir primeros 10 resultados
    for(int i = 0; i < 10; i++){
        std::cout << "c[" << i << "] = " << c[i] << "\n";
    }

    return 0;
}
