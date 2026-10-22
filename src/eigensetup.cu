__global__ void eigensetup(const float* __restrict__ cuda_ptr, float* __restrict__ lina_ptr) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadID >= ENGINE.n_threads) return;

    // unpack CUDA_MX.bin
    const float* row_startpos = &cuda_ptr[static_cast<size_t>(threadID)*ENGINE.cuda_mx_width];
    const int ns = ENGINE.statespace_dim, nd = ENGINE.discrete_dim, nt = ENGINE.total_dim;
    float K[MAX_STATESPACE_DIM] = {0.0f};
    #pragma unroll
    for (int i=0; i<MAX_STATESPACE_DIM; i++) K[i] = i < ns ? row_startpos[i] : 0.0f;
    float L[MAX_STATESPACE_DIM] = {0.0f};
    #pragma unroll
    for (int i=0; i<MAX_STATESPACE_DIM; i++) L[i] = i < ns ? row_startpos[ns + i] : 0.0f;

    // compute and store eigenvalues of linearized A
    // LIN_A = [ STATESPACE_A - STATESPACE_B*K - L*(STATESPACE_C - STATESPACE_D*K), L*DISCRETE_C ; -BD_U*K, AD ]
    float LIN_A1[MAX_STATESPACE_DIM*MAX_STATESPACE_DIM] = {0.0f};
    float LIN_A2[MAX_STATESPACE_DIM*MAX_DISCRETE_DIM] = {0.0f};
    float LIN_A3[MAX_DISCRETE_DIM*MAX_STATESPACE_DIM] = {0.0f};

    float plantb_k[MAX_STATESPACE_DIM*MAX_STATESPACE_DIM] = {0.0f};
    float plantd_k[MAX_STATESPACE_DIM] = {0.0f};
    float plantc_plantdk[MAX_STATESPACE_DIM] = {0.0f};
    float l_plantcplantdk[MAX_STATESPACE_DIM*MAX_STATESPACE_DIM] = {0.0f};
    float bdu_k[MAX_DISCRETE_DIM*MAX_STATESPACE_DIM] = {0.0f};

    matmul<MAX_STATESPACE_DIM, 1, MAX_STATESPACE_DIM>(STATESPACE_B,K,plantb_k);
    matmul<1, 1, MAX_STATESPACE_DIM>(STATESPACE_D,K,plantd_k);
    vecsub<MAX_STATESPACE_DIM>(STATESPACE_C,plantd_k,plantc_plantdk);
    matmul<MAX_STATESPACE_DIM, 1, MAX_STATESPACE_DIM>(L,plantc_plantdk,l_plantcplantdk);
    matmul<MAX_DISCRETE_DIM, 1, MAX_STATESPACE_DIM>(DISCRETE_BU,K,bdu_k);

    vecsub3<MAX_STATESPACE_DIM*MAX_STATESPACE_DIM>(STATESPACE_A,plantb_k,l_plantcplantdk,LIN_A1);
    matmul<MAX_STATESPACE_DIM, 1, MAX_DISCRETE_DIM>(L,DISCRETE_C,LIN_A2);
    #pragma unroll
    for (int i=0; i<MAX_DISCRETE_DIM*MAX_STATESPACE_DIM; i++) LIN_A3[i] = -bdu_k[i];

    // write the four blocks out as one compact row-major nt x nt matrix
    float* out = &lina_ptr[static_cast<size_t>(threadID)*ENGINE.lina_mx_width];
    #pragma unroll
    for (int i=0; i<MAX_STATESPACE_DIM; i++) {
        if (i >= ns) continue;
        #pragma unroll
        for (int j=0; j<MAX_STATESPACE_DIM; j++) if (j < ns) out[i*nt+j] = LIN_A1[i*MAX_STATESPACE_DIM+j];
        #pragma unroll
        for (int j=0; j<MAX_DISCRETE_DIM; j++) if (j < nd) out[i*nt+ns+j] = LIN_A2[i*MAX_DISCRETE_DIM+j];
    }

    #pragma unroll
    for (int i=0; i<MAX_DISCRETE_DIM; i++) {
        if (i >= nd) continue;
        #pragma unroll
        for (int j=0; j<MAX_STATESPACE_DIM; j++) if (j < ns) out[(ns+i)*nt+j] = LIN_A3[i*MAX_STATESPACE_DIM+j];
        #pragma unroll
        for (int j=0; j<MAX_DISCRETE_DIM; j++) if (j < nd) out[(ns+i)*nt+ns+j] = DISCRETE_A[i*MAX_DISCRETE_DIM+j];
    }
}
