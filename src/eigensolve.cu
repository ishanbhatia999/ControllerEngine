__host__ void eigensolve(const float* __restrict__ lina_ptr, float* eigen_ptr) {
    using MatD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, MAX_TOTAL_DIM, MAX_TOTAL_DIM>;
    using MapF = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
    const int nt = CFG.total_dim;

    #pragma omp parallel for
    for (int t=0; t<CFG.n_threads; t++) {
        // solve in fp64 for accuracy
        MapF Af(lina_ptr + static_cast<size_t>(t)*CFG.lina_mx_width, nt, nt);
        MatD A = Af.cast<double>();
        Eigen::EigenSolver<MatD> eigenA(A,false);
        const auto& eig = eigenA.eigenvalues();
        for (int i=0; i<nt; i++) {
            // store in fp32 for compatibility
            eigen_ptr[static_cast<size_t>(t)*CFG.eigen_mx_width + 2*i] = (float)eig[i].real();
            eigen_ptr[static_cast<size_t>(t)*CFG.eigen_mx_width + 2*i + 1] = (float)eig[i].imag();
        }
    }
}
