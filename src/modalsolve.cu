__host__ void modalsolve(const float* __restrict__ ata_ptr, const float* __restrict__ atb_ptr, const float* __restrict__ eigen_ptr, float* __restrict__ info_ptr) {
    constexpr int MAX_NC = 2*MAX_TOTAL_DIM;
    using MatF = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, 0, MAX_NC, MAX_NC>;
    using VecF = Eigen::Matrix<float, Eigen::Dynamic, 1, 0, MAX_NC, 1>;
    const int nt = CFG.total_dim, nc2 = 2*nt;
    const float TS = CFG.ts, BANDWIDTH = CFG.bandwidth, DAMPING = CFG.damping, OVERSHOOT = CFG.overshoot, SSE = CFG.steady_state_error;
    const float UNSTABLE_ERR = CFG.unstable_err, EIG_FAIL_ERR = CFG.eig_fail_err;

    #pragma omp parallel for
    for (int t=0; t<CFG.n_threads; t++) {
        float* out = &info_ptr[static_cast<size_t>(t)*CFG.info_mx_width];
        float* sc = out + CFG.info_scoring;     // score, behavior_err, margin_err
        float* fr = out + CFG.info_freqresp;    // aggregate wn, zeta, step overshoot and steady-state value (from modalsetup)
        float* md = out + CFG.info_modal;       // (wn,zeta) per mode, energy per mode, unique count
        if (sc[0] != 0.0f) continue;    // if it's flagged as unstable then skip

        // s = log(lambda)/TS, discrete --> continuous
        float Re[MAX_TOTAL_DIM] = {0.0f}; float Im[MAX_TOTAL_DIM] = {0.0f};
        const float* eig = &eigen_ptr[static_cast<size_t>(t)*CFG.eigen_mx_width];
        for (int i=0; i<nt; i++) {
            Re[i] = eig[2*i];
            Im[i] = eig[2*i + 1];
        }
        float max_mag = 0.0f;
        for (int i=0; i<nt; i++) {
            float re = Re[i], im = Im[i];
            const float mag = hypotf(re,im);
            max_mag = fmaxf(max_mag, mag);
            Re[i] = logf(fmaxf(mag, 0x1.a484cp-100))/TS;
            Im[i] = atan2f(im,re)/TS;
        }
        if (max_mag > 1.0f) { sc[0] = UNSTABLE_ERR + logf(max_mag)/TS; sc[1] = sc[0]; continue; }

        float _re = 0x1.2a05f2p+33f;
        float _im = 0x1.2a05f2p+33f;
        bool eig_states[MAX_TOTAL_DIM] = {false};
        int unique = nt;
        for (int i=0; i<nt; i++) {
            if (fabsf(Re[i]-_re) < 0x1.0c6f7ap-20f  &&  fabsf(fabsf(Im[i])-fabsf(_im)) < 0x1.0c6f7ap-20f) { eig_states[i] = true; unique--; }
            _re = Re[i];
            _im = Im[i];
        }
        float modal_response[2*MAX_TOTAL_DIM] = {0.0f};
        unique = 0;
        for (int i=0; i<nt; i++) {
            if (!eig_states[i]) {
                modal_response[2*unique] = Re[i];
                modal_response[2*unique+1] = fabsf(Im[i]);
                unique++;
            }
        }

        // LDLT solve, AtA already exists from modalsetup
        Eigen::Map<const Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> AtA(&ata_ptr[static_cast<size_t>(t)*CFG.ata_mx_width], nc2, nc2);
        Eigen::Map<const Eigen::Matrix<float,Eigen::Dynamic,1>> Atb(&atb_ptr[static_cast<size_t>(t)*CFG.atb_mx_width], nc2);

        // scale linear system to improve conditioning
        VecF col_scale(nc2);
        for (int i=0; i<nc2; i++) {
            float s = sqrtf(AtA(i,i));
            col_scale(i) = (s > 0x1.0c6f7ap-20f) ? s : 1.0f;
        }
        MatF AtA_scaled = AtA.selfadjointView<Eigen::Lower>();
        AtA_scaled = col_scale.cwiseInverse().asDiagonal() * AtA_scaled * col_scale.cwiseInverse().asDiagonal();
        VecF Atb_scaled = Atb.cwiseQuotient(col_scale);

        VecF coeffs = AtA_scaled.selfadjointView<Eigen::Lower>().ldlt().solve(Atb_scaled);
        coeffs = coeffs.cwiseQuotient(col_scale);

        // per-mode energy and then compute aggregate wn/zeta for oscillatory modes only and dominant oscillatory mode
        float energy_osc_sum = 0.0f, log_wn_weighted = 0.0f, zeta_weighted = 0.0f;
        for (int e=0; e<unique; e++) {
            int id = 2*e;
            const float c0 = coeffs(id), c1 = coeffs(id+1);
            const float g00 = AtA(id,id), g01 = AtA(id+1,id), g11 = AtA(id+1,id+1);
            const float energy = c0*c0*g00 + 2.0f*c0*c1*g01 + c1*c1*g11;
            const float re_e = modal_response[id], im_e = modal_response[id+1];
            const float wn_e = fmaxf(sqrtf(re_e*re_e + im_e*im_e), 0x1.0c6f7ap-20f);

            // debug output for all modes
            md[id] = wn_e;
            md[id+1] = -re_e/wn_e;
            md[2*nt+e] = energy;

            if (im_e < 0x1.0c6f7ap-20f) continue;    // we do not want a non-oscillatory mode
            energy_osc_sum += energy;
            log_wn_weighted += energy*logf(wn_e);
            zeta_weighted += energy*(-re_e/wn_e);
        }
        md[3*nt] = (float)unique;
        float wn, zeta;
        if (energy_osc_sum <= 0x1.0c6f7ap-20f) { wn = 0x1.0c6f7ap-20f; zeta = 0.0f; }    // no significant oscillatory content
        else { wn = expf(log_wn_weighted/energy_osc_sum); zeta = zeta_weighted/energy_osc_sum; }

        const float d2 = DAMPING*DAMPING;
        const float targetwn = 2.0f*PI*BANDWIDTH/sqrtf( (1.0f-2.0f*d2) + sqrtf(4.0f*d2*d2 - 4.0f*d2 + 2.0f) );
        float behavior_err = logf(wn/targetwn)*logf(wn/targetwn) + (zeta-DAMPING)*(zeta-DAMPING)/DAMPING/DAMPING;
        const float err_os = fmaxf(0.0f, fr[2] - OVERSHOOT)/OVERSHOOT;    // hinge: zero inside the overshoot spec
        behavior_err += err_os*err_os;
        const float err_ss = fmaxf(0.0f, fabsf(fr[3]/CFG.ref_amplitude - 1.0f) - SSE)/SSE;    // hinge on |yss/ref - 1|
        behavior_err += err_ss*err_ss;

        sc[0] = isfinite(behavior_err) ? behavior_err : EIG_FAIL_ERR;
        sc[1] = sc[0];
        fr[0] = wn;
        fr[1] = zeta;
    }
}
