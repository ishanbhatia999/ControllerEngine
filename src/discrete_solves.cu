template<ControllerType TYPE>
__global__ void discrete_solve(const float* __restrict__ cuda_ptr, const float* __restrict__ ref, float* __restrict__ yplant);

// runs the MAX_NONLIN_DIM nonlinear chains
__device__ __forceinline__ void nl_compute(const float* __restrict__ nlplant, float* __restrict__ W) {
    #pragma unroll
    for (int i=0; i<MAX_NONLIN_DIM; i++) W[i] = interpreter<MAX_NL_DIM, MAX_NL_INSTRUCTIONS, false>(&NL_OPS[i*MAX_NL_INSTRUCTIONS], nlplant);
}



template<>
__global__ void discrete_solve<ControllerType::Luenberger>(const float* __restrict__ cuda_ptr, const float* __restrict__ ref, float* __restrict__ yplant) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadID >= ENGINE.n_threads) return;

    // unpack CUDA_MX.bin
    const float* row_startpos = &cuda_ptr[static_cast<size_t>(threadID)*ENGINE.cuda_mx_width];
    const int n = ENGINE.statespace_dim;
    float K[MAX_STATESPACE_DIM] = {0.0f};
    #pragma unroll
    for (int i=0; i<MAX_STATESPACE_DIM; i++) K[i] = i < n ? row_startpos[i] : 0.0f;
    float L[MAX_STATESPACE_DIM] = {0.0f};
    #pragma unroll
    for (int i=0; i<MAX_STATESPACE_DIM; i++) L[i] = i < n ? row_startpos[n + i] : 0.0f;
    float N = row_startpos[n + n];

    // initialize state
    float xh_hi[MAX_STATESPACE_DIM] = {0.0f}, xh_lo[MAX_STATESPACE_DIM] = {0.0f};
    float xp_hi[MAX_DISCRETE_DIM] = {0.0f}, xp_lo[MAX_DISCRETE_DIM] = {0.0f};

    float* yrow = &yplant[static_cast<size_t>(threadID)*ENGINE.time];
    for (int t=0; t<ENGINE.time; t++) {
        // compute state y for observer error
        float y = interpreter<MAX_DISCRETE_DIM, MAX_DC_INSTRUCTIONS, false>(DC_OPS, xp_hi);
        yrow[t] = y;    // update yplant

        // compute controller input for plant
        float k_xhat = vecmul<MAX_STATESPACE_DIM>(K, xh_lo, vecmul<MAX_STATESPACE_DIM>(K, xh_hi, 0.0f));
        float u_raw = __fmaf_rn(N, ref[t], -k_xhat);
        float u = interpreter<MAX_STATESPACE_DIM, MAX_U_INSTRUCTIONS, true>(U_OPS, xh_hi, u_raw);

        // observer output error
        float yhat = __fmaf_rn(STATESPACE_D[0], u, vecmul<MAX_STATESPACE_DIM>(STATESPACE_C, xh_lo, vecmul<MAX_STATESPACE_DIM>(STATESPACE_C, xh_hi, 0.0f)));
        float yerr = y - yhat;

        // observer increment: (STATESPACE_A - I)*xhat + STATESPACE_B*u + L*yerr
        float dh[MAX_STATESPACE_DIM], dl[MAX_STATESPACE_DIM];
        #pragma unroll
        for (int i=0; i<MAX_STATESPACE_DIM; i++) {
            const float base = __fmaf_rn(L[i], yerr, STATESPACE_B[i]*u);
            dh[i] = vecmul<MAX_STATESPACE_DIM>(&STATESPACE_A_I[i*MAX_STATESPACE_DIM], xh_hi, base);
            dl[i] = vecmul<MAX_STATESPACE_DIM>(&STATESPACE_A_I[i*MAX_STATESPACE_DIM], xh_lo, 0.0f);
        }
        #pragma unroll
        for (int i=0; i<MAX_STATESPACE_DIM; i++) hladdf(xh_hi[i], xh_lo[i], dh[i], dl[i]);

        // define nonlinear terms
        float nlplant[MAX_NL_DIM];
        #pragma unroll
        for (int i=0; i<MAX_DISCRETE_DIM; i++) {
            nlplant[i] = xp_hi[i];
        } nlplant[MAX_DISCRETE_DIM] = y;

        float W[MAX_NONLIN_DIM] = {0.0f};
        nl_compute(nlplant, W);

        // plant increment: (DISCRETE_A - I)*xplant + DISCRETE_BU*u + DISCRETE_BW*W
        float ph[MAX_DISCRETE_DIM], pl[MAX_DISCRETE_DIM];
        #pragma unroll
        for (int i=0; i<MAX_DISCRETE_DIM; i++) {
            const float base = vecmul<MAX_NONLIN_DIM>(&DISCRETE_BW[i*MAX_NONLIN_DIM], W, DISCRETE_BU[i]*u);
            ph[i] = vecmul<MAX_DISCRETE_DIM>(&DISCRETE_A_I[i*MAX_DISCRETE_DIM], xp_hi, base);
            pl[i] = vecmul<MAX_DISCRETE_DIM>(&DISCRETE_A_I[i*MAX_DISCRETE_DIM], xp_lo, 0.0f);
        }
        #pragma unroll
        for (int i=0; i<MAX_DISCRETE_DIM; i++) hladdf(xp_hi[i], xp_lo[i], ph[i], pl[i]);
    }
}



template<>
__global__ void discrete_solve<ControllerType::Kalman>(const float* __restrict__ cuda_ptr, const float* __restrict__ ref, float* __restrict__ yplant) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadID >= ENGINE.n_threads) return;

    // unpack CUDA_MX.bin
    const float* row_startpos = &cuda_ptr[static_cast<size_t>(threadID)*ENGINE.cuda_mx_width];
    const int n = ENGINE.statespace_dim;
    float K[MAX_STATESPACE_DIM] = {0.0f};
    #pragma unroll
    for (int i=0; i<MAX_STATESPACE_DIM; i++) K[i] = i < n ? row_startpos[i] : 0.0f;
    float N = row_startpos[n];
    float R = row_startpos[n + 1];
    float Q[MAX_STATESPACE_DIM] = {0.0f};
    #pragma unroll
    for (int i=0; i<MAX_STATESPACE_DIM; i++) Q[i] = i < n ? row_startpos[n + 2 + i] : 0.0f;

    // initialize state
    float xh_hi[MAX_STATESPACE_DIM] = {0.0f}, xh_lo[MAX_STATESPACE_DIM] = {0.0f};
    float xp_hi[MAX_DISCRETE_DIM] = {0.0f}, xp_lo[MAX_DISCRETE_DIM] = {0.0f};
    float P[MAX_STATESPACE_DIM*MAX_STATESPACE_DIM] = {0.0f}, KG[MAX_STATESPACE_DIM] = {0.0f};
    float PHT[MAX_STATESPACE_DIM] = {0.0f}, HPHR = 0.0f;
    float IKHP[MAX_STATESPACE_DIM*MAX_STATESPACE_DIM] = {0.0f}, KIRK[MAX_STATESPACE_DIM*MAX_STATESPACE_DIM] = {0.0f};

    float* yrow = &yplant[static_cast<size_t>(threadID)*ENGINE.time];
    for (int t=0; t<ENGINE.time; t++) {
        // compute state y for observer error
        float y = interpreter<MAX_DISCRETE_DIM, MAX_DC_INSTRUCTIONS, false>(DC_OPS, xp_hi);
        yrow[t] = y;    // update yplant

        // compute controller input for plant
        float k_xhat = vecmul<MAX_STATESPACE_DIM>(K, xh_lo, vecmul<MAX_STATESPACE_DIM>(K, xh_hi, 0.0f));
        float u_raw = __fmaf_rn(N, ref[t], -k_xhat);
        float u = interpreter<MAX_STATESPACE_DIM, MAX_U_INSTRUCTIONS, true>(U_OPS, xh_hi, u_raw);

        // Kalman state update: (STATESPACE_A - I)*xhat + STATESPACE_B*u
        float dh[MAX_STATESPACE_DIM] = {0.0f}, dl[MAX_STATESPACE_DIM] = {0.0f};
        #pragma unroll
        for (int i=0; i<MAX_STATESPACE_DIM; i++) {
            dh[i] = vecmul<MAX_STATESPACE_DIM>(&STATESPACE_A_I[i*MAX_STATESPACE_DIM], xh_hi, STATESPACE_B[i]*u);
            dl[i] = vecmul<MAX_STATESPACE_DIM>(&STATESPACE_A_I[i*MAX_STATESPACE_DIM], xh_lo, 0.0f);
        }
        #pragma unroll
        for (int i=0; i<MAX_STATESPACE_DIM; i++) hladdf(xh_hi[i], xh_lo[i], dh[i], dl[i]);

        // estimate covariance initial update: P = A*P*A^T + Q
        ABATf<MAX_STATESPACE_DIM,MAX_STATESPACE_DIM>(STATESPACE_A,P,P);
        #pragma unroll
        for (int i=0; i<MAX_STATESPACE_DIM; i++) P[i*MAX_STATESPACE_DIM+i] = __fadd_rn(P[i*MAX_STATESPACE_DIM+i],Q[i]);

        // solve for Kalman gain
        ABTf<MAX_STATESPACE_DIM,MAX_STATESPACE_DIM,1>(P,STATESPACE_C,PHT);
        ABATf<1,MAX_STATESPACE_DIM>(STATESPACE_C,P,&HPHR);
        HPHR = __fdividef(1.0f,__fadd_rn(HPHR,R));
        #pragma unroll
        for (int i=0; i<MAX_STATESPACE_DIM; i++) KG[i] = __fmul_rn(PHT[i], HPHR);

        // Kalman state update: xhat += KG*(z-STATESPACE_C*xhat);
        float dzh = __fmul_rn(-1.0f,vecmul<MAX_STATESPACE_DIM>(STATESPACE_C, xh_hi, -y));
        float dzl = __fmul_rn(-1.0f,vecmul<MAX_STATESPACE_DIM>(STATESPACE_C, xh_lo, 0.0f));
        float s = 0.0f, e = 0.0f;
        #pragma unroll
        for (int i=0; i<MAX_STATESPACE_DIM; i++) {
            hlprodf(KG[i],dzh,s,e);
            e = __fmaf_rn(KG[i],dzl,e);
            hladdf(xh_hi[i],xh_lo[i],s,e);
        }

        // update covariance matrix
        matmul<MAX_STATESPACE_DIM,1,MAX_STATESPACE_DIM>(KG,STATESPACE_C,IKHP);
        #pragma unroll
        for (int i=0; i<MAX_STATESPACE_DIM; i++) IKHP[i*MAX_STATESPACE_DIM+i] = __fsub_rn(IKHP[i*MAX_STATESPACE_DIM+i],1.0f);
        #pragma unroll
        for (int i=0; i<MAX_STATESPACE_DIM; i++) {
            #pragma unroll
            for (int j=0; j<MAX_STATESPACE_DIM; j++) IKHP[i*MAX_STATESPACE_DIM+j] = -IKHP[i*MAX_STATESPACE_DIM+j];
        }
        ABATf<MAX_STATESPACE_DIM,MAX_STATESPACE_DIM>(IKHP,P,IKHP);
        ABATf<MAX_STATESPACE_DIM,1>(KG,&R,KIRK);
        matadd<MAX_STATESPACE_DIM,MAX_STATESPACE_DIM>(IKHP,KIRK,P);

        // define nonlinear terms
        float nlplant[MAX_NL_DIM];
        #pragma unroll
        for (int i=0; i<MAX_DISCRETE_DIM; i++) {
            nlplant[i] = xp_hi[i];
        } nlplant[MAX_DISCRETE_DIM] = y;

        float W[MAX_NONLIN_DIM] = {0.0f};
        nl_compute(nlplant, W);

        // plant increment: (DISCRETE_A - I)*xplant + DISCRETE_BU*u + DISCRETE_BW*W
        float ph[MAX_DISCRETE_DIM], pl[MAX_DISCRETE_DIM];
        #pragma unroll
        for (int i=0; i<MAX_DISCRETE_DIM; i++) {
            const float base = vecmul<MAX_NONLIN_DIM>(&DISCRETE_BW[i*MAX_NONLIN_DIM], W, DISCRETE_BU[i]*u);
            ph[i] = vecmul<MAX_DISCRETE_DIM>(&DISCRETE_A_I[i*MAX_DISCRETE_DIM], xp_hi, base);
            pl[i] = vecmul<MAX_DISCRETE_DIM>(&DISCRETE_A_I[i*MAX_DISCRETE_DIM], xp_lo, 0.0f);
        }
        #pragma unroll
        for (int i=0; i<MAX_DISCRETE_DIM; i++) hladdf(xp_hi[i], xp_lo[i], ph[i], pl[i]);
    }
}

