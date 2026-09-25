#include "common.hpp"
#include <iostream>
#include <stdexcept>

static double dot_naive(const std::vector<double>& x,
                        const std::vector<double>& y) {
    double acc = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        acc += x[i] * y[i];
    return acc;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <n> [seed]\n";
        return 1;
    }
    std::size_t n  = std::stoull(argv[1]);
    unsigned   seed = (argc >= 3) ? static_cast<unsigned>(std::stoul(argv[2])) : 42u;

    auto x = make_array(n);
    auto y = make_array(n);
    fill_random(x, seed);
    fill_random(y, seed + 1);

    Timer t;
    t.start();
    double result = dot_naive(x, y);
    double elapsed = t.stop();

    sink(result);
    std::cout << "dot_naive n=" << n
              << " time_s="    << elapsed
              << " checksum="  << result << '\n';
    return 0;
}
