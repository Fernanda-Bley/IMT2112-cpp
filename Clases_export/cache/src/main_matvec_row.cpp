#include "common.hpp"
#include <iostream>

// y[i] = sum_j A[i*n+j] * x[j]  — loop externo i, interno j
static void matvec_row(const std::vector<double>& A,
                       const std::vector<double>& x,
                       std::vector<double>&       y,
                       std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        double acc = 0.0;
        for (std::size_t j = 0; j < n; ++j)
            acc += A[i * n + j] * x[j];
        y[i] = acc;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <n> [seed]\n";
        return 1;
    }
    std::size_t n   = std::stoull(argv[1]);
    unsigned   seed = (argc >= 3) ? static_cast<unsigned>(std::stoul(argv[2])) : 42u;

    auto A = make_matrix(n);
    auto x = make_array(n);
    auto y = make_array(n);
    fill_random(A, seed);
    fill_random(x, seed + 1);

    Timer t;
    t.start();
    matvec_row(A, x, y, n);
    double elapsed = t.stop();

    double checksum = 0.0;
    for (double v : y) checksum += v;
    sink(checksum);

    std::cout << "matvec_row n=" << n
              << " time_s="      << elapsed
              << " checksum="    << checksum << '\n';
    return 0;
}
