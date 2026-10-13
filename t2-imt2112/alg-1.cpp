

for(int i = 0; i < N; ++i) {
    for(int k = 0; k < N; ++k) {
        for(int j = 0; j < N; ++j) {
            C[i*N + j] += A[i*N + k] * B[k*N + j];
        }
    }
}