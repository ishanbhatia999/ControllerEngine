#if !defined(__CUDA_ARCH__) 
template<int NS>
__host__ void allmargin_fixed(const float* __restrict__ lina_ptr, const float* __restrict__ cuda_ptr, const std::complex<float>* __restrict__ z_ptr, float* __restrict__ info_ptr) {
    using MatC = Eigen::Matrix<std::complex<float>, NS, NS>;
    using VecC = Eigen::Matrix<std::complex<float>, NS, 1>;
    const int nt = CFG.total_dim;
    const int FREQ = CFG.freq;
    const float TS = CFG.ts, TARGET_GM = CFG.target_gm, TARGET_PM = CFG.target_pm;
    const float UNSTABLE_ERR = CFG.unstable_err, EIG_FAIL_ERR = CFG.eig_fail_err;

    // margins (batch: observer's nominal model. serial: true linear plant)
    Eigen::Map<const Eigen::Matrix<float, NS, NS, Eigen::RowMajor>> PLANT_A(CFG.serial ? DISCRETE_A_HOST : STATESPACE_A_HOST);
    Eigen::Map<const Eigen::Matrix<float, NS, 1>> PLANT_B(CFG.serial ? DISCRETE_BU_HOST : STATESPACE_B_HOST);
    Eigen::Map<const Eigen::Matrix<float, NS, 1>> PLANT_C(CFG.serial ? DISCRETE_C_HOST : STATESPACE_C_HOST);
    const float PLANT_D = CFG.serial ? 0.0f : STATESPACE_D_HOST[0];
    const MatC PLANT_A_c = PLANT_A.template cast<std::complex<float>>();
    const VecC PLANT_B_c = PLANT_B.template cast<std::complex<float>>();
    const VecC PLANT_C_c = PLANT_C.template cast<std::complex<float>>();

    std::vector<std::complex<float>> plant_resp(FREQ);
    std::vector<bool> plant_ok(FREQ);
    {
        const MatC I_plant = MatC::Identity();
        for (int w=0; w<FREQ; w++) {
            const MatC M = z_ptr[w]*I_plant - PLANT_A_c;
            const VecC x = M.partialPivLu().solve(PLANT_B_c);
            plant_ok[w] = x.allFinite();
            plant_resp[w] = plant_ok[w] ? ((PLANT_C_c.transpose()*x).value() + PLANT_D) : std::complex<float>(0.0f, 0.0f);
        }
    }

    #pragma omp parallel for
    for (int t=0; t<CFG.n_threads; t++) {
        float* out = &info_ptr[static_cast<size_t>(t)*CFG.info_mx_width];
        float* sc = out + CFG.info_scoring;      // score, behavior_err, margin_err
        float* am = out + CFG.info_allmargin;    // GM_dB, PM, DM_ms, stable
        if (sc[0] >= UNSTABLE_ERR) { sc[1] = sc[0]; sc[2] = sc[0]; am[0] = 0.0f; am[1] = 0.0f; am[2] = 0.0f; am[3] = 0.0f; continue; }    // set penalty score if it's flagged as unstable

        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> LIN_A1(&lina_ptr[static_cast<size_t>(t)*CFG.lina_mx_width], nt, nt);
        Eigen::Matrix<float, NS, NS> OPEN_A = LIN_A1.template topLeftCorner<NS, NS>();

        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 1>> row(&cuda_ptr[static_cast<size_t>(t)*CFG.cuda_mx_width], CFG.cuda_mx_width);
        Eigen::Matrix<float, NS, 1> K = row.template segment<NS>(0);
        Eigen::Matrix<float, NS, 1> L = row.template segment<NS>(NS);

        Eigen::Matrix<float, NS, 1> OPEN_B = L;
        Eigen::Matrix<float, NS, 1> OPEN_C = K;
        float OPEN_D = 0.0f;

        // check allmargin on closed loop w/ statespace plant
        using MatCL = Eigen::Matrix<float, 2*NS, 2*NS>;
        MatCL Acl_nom;
        Acl_nom.template topLeftCorner<NS, NS>()     = OPEN_A - PLANT_D*(OPEN_B*OPEN_C.transpose());
        Acl_nom.template topRightCorner<NS, NS>()    = -(OPEN_B*PLANT_C.transpose());
        Acl_nom.template bottomLeftCorner<NS, NS>()  = PLANT_B*OPEN_C.transpose();
        Acl_nom.template bottomRightCorner<NS, NS>() = PLANT_A;

        // solve in fp64 for accuracy: with |K| ~1e6 the fp32 solve calls loops unstable that eigensolve's fp64 solve
        // of the same closed loop (LIN_A, a similarity transform of this one in serial mode) finds stable
        using MatCLd = Eigen::Matrix<double, 2*NS, 2*NS>;
        Eigen::EigenSolver<MatCLd> es(Acl_nom.template cast<double>(), false);
        bool nominal_stable = es.info() == Eigen::Success;
        if (nominal_stable) {
            const auto& eig_nom = es.eigenvalues();
            for (int i=0; i<2*NS; i++) {
                if (std::abs(eig_nom[i]) >= 1.0) { nominal_stable = false; break; }
            }
        }
        if (!nominal_stable) {    // if the loop is unstable just penalize the score
            sc[1] = sc[0];
            sc[2] = UNSTABLE_ERR;
            am[0] = 0.0f; am[1] = 0.0f; am[2] = 0.0f; am[3] = 0.0f;
            float total = logf(sc[0] + UNSTABLE_ERR + 0x1.a484cp-100f);
            sc[0] = isfinite(total) ? total : EIG_FAIL_ERR;
            continue;
        }

        const MatC OPEN_A_c = OPEN_A.template cast<std::complex<float>>();
        const VecC OPEN_B_c = OPEN_B.template cast<std::complex<float>>();
        const VecC OPEN_C_c = OPEN_C.template cast<std::complex<float>>();
        const MatC I = MatC::Identity();

        // score on smallest margin (it is the least stable)
        float PM = 0x1.fffffep127f, GM_dB = 0x1.fffffep127f, DM_ms = 0x1.fffffep127f;
        bool gain_crossed = false;
        bool ok = true;
        float prevMag = 0.0f, prevPhase = 0.0f, prevRaw = 0.0f, prevTheta = 0.0f;

        for (int w=0; w<FREQ; w++) {
            if (!plant_ok[w]) { ok = false; break; }

            const MatC M = z_ptr[w]*I - OPEN_A_c;    // (z_ptr[w]*I - OPEN_A) x = OPEN_B
            const VecC x = M.partialPivLu().solve(OPEN_B_c);
            if (!x.allFinite()) { ok = false; break; }
            const std::complex<float> Gz = (OPEN_C_c.transpose()*x).value() + OPEN_D;    // G(z) = OPEN_C * x + OPEN_D

            const std::complex<float> resp = plant_resp[w]*Gz;    // loop(z) = plant(z)*compensator(z)
            const float m = std::abs(resp);
            const float raw = std::arg(resp);
            const float theta_w = std::arg(z_ptr[w]);

            if (w > 0) {
                float delta = raw - prevRaw;
                if (delta > PI) delta -= 2.0f*PI;
                else if (delta < -PI) delta += 2.0f*PI;
                float ph = prevPhase + delta;

                // gain crossover, wrap to (-180,180]
                if ((prevMag-1.0f)*(m-1.0f) <= 0.0f && m != prevMag) {
                    float frac = (1.0f - prevMag)/(m - prevMag);
                    float phase_c = prevPhase + frac*(ph-prevPhase);
                    gain_crossed = true;
                    float pm_deg = 180.0f + phase_c*(180.0f/PI);
                    pm_deg -= 360.0f*floorf((pm_deg + 180.0f)/360.0f);
                    PM = fminf(PM, pm_deg);

                    // delay margin at the gain crossover
                    float theta_c = prevTheta + frac*(theta_w-prevTheta);
                    float pm_rad_c = pm_deg*(PI/180.0f);
                    if (theta_c > 0x1.0c6f7ap-20f && pm_rad_c >= 0.0f) {
                        DM_ms = fminf(DM_ms, (pm_rad_c/theta_c)*TS*1000.0f);
                    }
                }
                // phase crossover
                {
                    float a = prevPhase + PI, b = ph + PI;
                    int ka = (int)floorf(a/(2.0f*PI)), kb = (int)floorf(b/(2.0f*PI));
                    if (ka != kb && b != a) {
                        float target = 2.0f*PI*(float)(b > a ? kb : ka);
                        float frac = (target - a)/(b - a);
                        float mag_p = prevMag + frac*(m-prevMag);
                        GM_dB = fminf(GM_dB, -20.0f*log10f(mag_p));
                    }
                }

                prevPhase = ph;
            } else {
                prevPhase = raw;
            }

            prevMag = m; prevRaw = raw; prevTheta = theta_w;
        }

        if (!ok) {
            sc[1] = sc[0]; sc[2] = EIG_FAIL_ERR; sc[0] = EIG_FAIL_ERR;
            am[0] = 0.0f; am[1] = 0.0f; am[2] = 0.0f; am[3] = 1.0f;
            continue;
        }

        float err_gm = fmaxf(0.0f, TARGET_GM - GM_dB)/TARGET_GM;
        float err_pm = gain_crossed ? fmaxf(0.0f, TARGET_PM - PM)/TARGET_PM : 0.0f;
        float margin_err = err_gm*err_gm + err_pm*err_pm;

        sc[1] = sc[0];
        sc[2] = margin_err;
        am[0] = GM_dB;
        am[1] = PM;
        am[2] = DM_ms;
        am[3] = 1.0f;
        float total = logf(sc[0] + margin_err + 0x1.a484cp-100f);
        sc[0] = isfinite(total) ? total : EIG_FAIL_ERR;
    }
}

__host__ void allmargin(const float* __restrict__ lina_ptr, const float* __restrict__ cuda_ptr, const std::complex<float>* __restrict__ z_ptr, float* __restrict__ info_ptr) {
    static_assert(MAX_STATESPACE_DIM == 8, "add/remove allmargin_fixed cases to match MAX_STATESPACE_DIM");
    switch (CFG.statespace_dim) {
        case 1: allmargin_fixed<1>(lina_ptr, cuda_ptr, z_ptr, info_ptr); break;
        case 2: allmargin_fixed<2>(lina_ptr, cuda_ptr, z_ptr, info_ptr); break;
        case 3: allmargin_fixed<3>(lina_ptr, cuda_ptr, z_ptr, info_ptr); break;
        case 4: allmargin_fixed<4>(lina_ptr, cuda_ptr, z_ptr, info_ptr); break;
        case 5: allmargin_fixed<5>(lina_ptr, cuda_ptr, z_ptr, info_ptr); break;
        case 6: allmargin_fixed<6>(lina_ptr, cuda_ptr, z_ptr, info_ptr); break;
        case 7: allmargin_fixed<7>(lina_ptr, cuda_ptr, z_ptr, info_ptr); break;
        case 8: allmargin_fixed<8>(lina_ptr, cuda_ptr, z_ptr, info_ptr); break;
        default: CUDAERR("allmargin has no instantiation for statespace_dim {}", CFG.statespace_dim);
    }
}
#else
__host__ void allmargin(const float* __restrict__ lina_ptr, const float* __restrict__ cuda_ptr, const std::complex<float>* __restrict__ z_ptr, float* __restrict__ info_ptr);
#endif
