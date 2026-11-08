__global__ void modalsetup(const float* __restrict__ yplant, const float* __restrict__ eigen_ptr, float* __restrict__ ata_ptr, float* __restrict__ atb_ptr, float* __restrict__ info_ptr) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadID >= ENGINE.n_threads) return;

    const int nt = ENGINE.total_dim;
    const float* yrow = &yplant[static_cast<size_t>(threadID)*ENGINE.time];

    // s = log(lambda)/TS, discrete --> continuous
    float Re[MAX_TOTAL_DIM] = {0.0f}; float Im[MAX_TOTAL_DIM] = {0.0f};
    const float* eig = &eigen_ptr[static_cast<size_t>(threadID)*ENGINE.eigen_mx_width];
    for (int i=0; i<nt; i++) {
        Re[i] = eig[2*i];
        Im[i] = eig[2*i + 1];
    }
    float max_mag = 0.0f;
    for (int i=0; i<nt; i++) {
        float re = Re[i], im = Im[i];
        const float mag = hypotf(re,im);
        max_mag = fmaxf(max_mag, mag);
        Re[i] = logf(fmaxf(mag, 0x1.a484cp-100))/ENGINE.ts;
        Im[i] = atan2f(im,re)/ENGINE.ts;
    }

    float* out = &info_ptr[static_cast<size_t>(threadID)*ENGINE.info_mx_width];
    float* sc = out + ENGINE.info_scoring;    // score, behavior_err, margin_err
    if (max_mag > 1.0f) { sc[0] = ENGINE.unstable_err + logf(max_mag)/ENGINE.ts; sc[1] = sc[0]; return; }

    // construct modal response
    float _re = 0x1.2a05f2p+33f;
    float _im = 0x1.2a05f2p+33f;
    bool eig_states[MAX_TOTAL_DIM];
    int unique = nt;
    for (int i=0; i<nt; i++) eig_states[i] = false;
    for (int i=0; i<nt; i++) {
        if (fabsf(Re[i]-_re) < 0x1.0c6f7ap-20f  &&  fabsf(fabsf(Im[i])-fabsf(_im)) < 0x1.0c6f7ap-20f) { eig_states[i] = true; unique--; }
        _re = Re[i];
        _im = Im[i];
    }
    float modal_response[2*MAX_TOTAL_DIM] = {0.0f};    // a,b, will solve for C1/C2 using least-sqares
    unique = 0;
    for (int i=0; i<nt; i++) {
        if (!eig_states[i]) {
            modal_response[2*unique] = Re[i];
            modal_response[2*unique+1] = fabsf(Im[i]);
            unique++;
        }
    }

    float shift[MAX_TOTAL_DIM] = {0.0f};
    for (int e=0; e<unique; e++) shift[e] = fmaxf(modal_response[2*e]*ENGINE.fit_t0, modal_response[2*e]*ENGINE.fit_t1);

    // steady-state offset so you can properly fit a sinusoid
    const int STEADY_START_ID = ENGINE.steady_start_id;
    float yss = 0.0f;
    for (int t=STEADY_START_ID; t<ENGINE.time; t++) yss += yrow[t];
    yss /= (float)(ENGINE.time - STEADY_START_ID);

    // step overshoot relative to the response's own final value; y·yss covers either step sign, y = 0 before the step
    float peak = 0.0f;
    for (int t=0; t<ENGINE.time; t++) peak = fmaxf(peak, yrow[t]*yss);
    out[ENGINE.info_freqresp + 2] = peak/fmaxf(yss*yss, 0x1.0c6f7ap-20f) - 1.0f;
    out[ENGINE.info_freqresp + 3] = yss;    // modalsolve scores it against the reference

    // AtA x = Atb, solve for x (modal coefficients)
    const int FIT_START_ID = ENGINE.fit_start_id;
    const int w = ENGINE.atb_mx_width;
    float AtA[2*MAX_TOTAL_DIM*2*MAX_TOTAL_DIM] = {0.0f};
    float Atb[2*MAX_TOTAL_DIM] = {0.0f};
    float coeffs[2*MAX_TOTAL_DIM] = {0.0f};
    float decay = 0.0f;
    float y = 0.0f;
    float tval = 0.0f;
    int id = 0;
    const int nc = 2*unique;
    const float fit_len = (ENGINE.fit_end-ENGINE.fit_start)*ENGINE.time;
    for (int t=0; t<fit_len; t++) {
        tval = static_cast<float>(FIT_START_ID+t)*ENGINE.ts;
        for (int e=0; e<unique; e++) {
            id = 2*e;
            decay = expf(modal_response[id]*tval - shift[e]);
            coeffs[id] = decay*cosf(modal_response[id+1]*tval);
            coeffs[id+1] = decay*sinf(modal_response[id+1]*tval);
        }
        y = yrow[FIT_START_ID+t] - yss;
        // AtA is symmetric so we only compute the lower triangle
        for (int i=0; i<nc; i++) {
            for (int j=0; j<=i; j++) {
                AtA[i*w+j] += coeffs[i]*coeffs[j];
            } Atb[i] += coeffs[i]*y;
        }
    }

    float* AtA_write = &ata_ptr[static_cast<size_t>(threadID)*ENGINE.ata_mx_width];
    float* Atb_write = &atb_ptr[static_cast<size_t>(threadID)*ENGINE.atb_mx_width];
    for (int i=0; i<ENGINE.ata_mx_width; i++) AtA_write[i] = AtA[i];
    for (int i=0; i<ENGINE.atb_mx_width; i++) Atb_write[i] = Atb[i];
}
