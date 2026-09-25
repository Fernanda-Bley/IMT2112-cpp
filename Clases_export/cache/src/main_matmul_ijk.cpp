#include "common.hpp"
#include <iostream>

// C = A*B  —  orden ijk
static void matmul_ijk(const std::vector<double>& A,
                       const std::vector<double>& B,
                       std::vector<double>&       C,
                       std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            C[i * n + j] = 0.0;

    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t k = 0; k < n; ++k)
                C[i * n + j] += A[i * n + k] * B[k * n + j];
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <n> [seed]\n";
        return 1;
    }
    std::size_t n   = std::stoull(argv[1]);
    unsigned   seed = (argc >= 3) ? static_cast<unsigned>(std::stoul(argv[2])) : 42u;

    auto A = make_matrix(n);
    auto B = make_matrix(n);
    auto C = make_matrix(n);
    fill_random(A, seed);
    fill_random(B, seed + 1);

    Timer t;
    t.start();
    matmul_ijk(A, B, C, n);
    double elapsed = t.stop();

    double checksum = 0.0;
    for (double v : C) checksum += v;
    sink(checksum);

    std::cout << "matmul_ijk n=" << n
              << " time_s="      << elapsed
              << " checksum="    << checksum << '\n';
    return 0;
}
