#pragma once
#include <chrono>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Allocation — matrices stored row-major, flattened (size n*n)
// ---------------------------------------------------------------------------

inline std::vector<double> make_array(std::size_t n) {
    return std::vector<double>(n, 0.0);
}

inline std::vector<double> make_matrix(std::size_t n) {
    return std::vector<double>(n * n, 0.0);
}

// ---------------------------------------------------------------------------
// Random fill
// ---------------------------------------------------------------------------

inline void fill_random(std::vector<double>& a, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (auto& v : a)
        v = dist(rng);
}

// ---------------------------------------------------------------------------
// Timer
// ---------------------------------------------------------------------------

struct Timer {
    using Clock = std::chrono::steady_clock;
    Clock::time_point t0;

    void start() { t0 = Clock::now(); }

    double stop() const {
        return std::chrono::duration<double>(Clock::now() - t0).count();
    }
};

// ---------------------------------------------------------------------------
// Sink: prevents dead-code elimination of kernel results
// ---------------------------------------------------------------------------

inline void sink(double x) {
    asm volatile("" : : "g"(x) : "memory");
}
