#include <omp.h>
#include <iostream>

int main(){
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int p  = omp_get_num_threads();
        std::cout << "Hola desde el hilo " << id
                  << " de " << p << "\n";
    }
    return 0;
}
