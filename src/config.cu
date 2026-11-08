// ControllerEngine.config: one `key = value` per line, `#` starts a comment, lists are comma separated.
// Read fresh on every RunControllerEngine() call, so nothing in it needs a rebuild -- only the MAX_*
// limits in main.cu do.

struct EngineConfig {
    // sizing (actual sizes, zero-padded up to the MAX_* limits on the device)
    int statespace_dim, discrete_dim, nonlin_dim, n_params, input_size;
    int dc_instructions, u_instructions;
    int nl_instructions[MAX_NONLIN_DIM];

    // launch
    int n_cores, n_rounds;
    bool serial;    // mode = serial: one plant per round (stacked DISCRETE_*/BASE_PARAMS .bin), margins against that plant

    // simulation
    int time;    // samples per run
    float duration;    // seconds per run, so ts = duration/time
    double ref_step_time;    // fraction of the run; reference is 0 before it, ref_amplitude from it on
    float ref_amplitude;

    // scoring / fit
    float fit_start, fit_end, steady_start;
    float bandwidth, damping, overshoot, steady_state_error, crossover_coeff, target_gm, target_pm, target_b;
    float eig_fail_err, unstable_err;
    int freq;
    float freq_min;
    bool dump_diagnostics;

    // derived by derive(), never read from the file
    int n_threads;    // candidates per round, one full-occupancy wave: n_cores / # cores per SM * # threads per SM
    float freq_max;    // Nyquist, pi/TS
    int total_dim, dc_dim, u_dim, nl_dim;
    int cuda_mx_width, info_scoring, info_freqresp, info_allmargin, info_modal, info_mx_width;
    int eigen_mx_width, lina_mx_width, ata_mx_width, atb_mx_width;
    int nl_instructions_total;
    float ts, fit_t0, fit_t1;
    float damping2, damping4, crossover;
    size_t total_candidates, cuda_mx_size, info_mx_size, eigen_mx_size, lina_mx_size, ata_mx_size, atb_mx_size, yplant_size;

    void derive() {
        n_threads = (n_cores/128)*1536;
        ts = duration/time;
        freq_max = PI/ts;
        total_dim = statespace_dim + discrete_dim;
        dc_dim = discrete_dim;
        u_dim = statespace_dim;
        nl_dim = discrete_dim+1;
        nl_instructions_total = 0;
        for (int i=0; i<nonlin_dim; i++) nl_instructions_total += nl_instructions[i];

        cuda_mx_width = 2*statespace_dim+1;
        info_scoring = input_size;
        info_freqresp = info_scoring+3;
        info_allmargin = info_freqresp+4;
        info_modal = info_allmargin+5;
        info_mx_width = info_modal+3*total_dim+1;
        eigen_mx_width = 2*total_dim;
        lina_mx_width = total_dim*total_dim;
        ata_mx_width = 2*total_dim*2*total_dim;
        atb_mx_width = 2*total_dim;

        fit_t0 = fit_start*duration;
        fit_t1 = (fit_end*time - 1.0f)*ts;
        
        damping2 = damping*damping; damping4 = damping2*damping2;
        // rad/s: wn from the bandwidth/damping spec (same as modalsolve's targetwn), times the 2nd-order gain crossover ratio wc/wn
        const float wn = 2.0f*PI*bandwidth/sqrtf( (1.0f-2.0f*damping2) + sqrtf(4.0f*damping4 - 4.0f*damping2 + 2.0f) );
        crossover = crossover_coeff*wn*sqrtf( sqrtf(4.0f*damping4 + 1.0f) - 2.0f*damping2 );

        total_candidates = static_cast<size_t>(n_rounds) * n_threads;
        cuda_mx_size = static_cast<size_t>(n_threads) * cuda_mx_width;
        info_mx_size = static_cast<size_t>(n_threads) * info_mx_width;
        eigen_mx_size = static_cast<size_t>(n_threads) * eigen_mx_width;
        lina_mx_size = static_cast<size_t>(n_threads) * lina_mx_width;
        ata_mx_size = static_cast<size_t>(n_threads) * ata_mx_width;
        atb_mx_size = static_cast<size_t>(n_threads) * atb_mx_width;
        yplant_size = static_cast<size_t>(n_threads) * time;
    }
};

EngineConfig CFG;

// the slice of CFG the kernels read, kept in __constant__ memory
struct EngineParams {
    int n_threads, time;
    int statespace_dim, discrete_dim, total_dim;
    int cuda_mx_width, info_mx_width, info_scoring, info_freqresp;
    int eigen_mx_width, lina_mx_width, ata_mx_width, atb_mx_width;
    int fit_start_id, steady_start_id;
    float ts, fit_start, fit_end, fit_t0, fit_t1, unstable_err;
};

__host__ EngineParams make_engine_params(const EngineConfig& c) {
    EngineParams p;
    p.n_threads = c.n_threads; p.time = c.time;
    p.statespace_dim = c.statespace_dim; p.discrete_dim = c.discrete_dim; p.total_dim = c.total_dim;
    p.cuda_mx_width = c.cuda_mx_width; p.info_mx_width = c.info_mx_width; p.info_scoring = c.info_scoring; p.info_freqresp = c.info_freqresp;
    p.eigen_mx_width = c.eigen_mx_width; p.lina_mx_width = c.lina_mx_width; p.ata_mx_width = c.ata_mx_width; p.atb_mx_width = c.atb_mx_width;
    p.fit_start_id = (int)(c.fit_start*c.time);
    p.steady_start_id = (int)(c.steady_start*c.time);
    p.ts = c.ts; p.fit_start = c.fit_start; p.fit_end = c.fit_end; p.fit_t0 = c.fit_t0; p.fit_t1 = c.fit_t1;
    p.unstable_err = c.unstable_err;
    return p;
}

__host__ EngineConfig load_config(const std::string& filename) {
    std::ifstream file(filename); if (!file) CUDAERR("Cannot open {}", filename);

    std::map<std::string, std::string> kv;
    std::string line;
    int lineno = 0;
    auto trim = [](std::string s) {
        const size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
        return a == std::string::npos ? std::string() : s.substr(a, b-a+1);
    };
    while (std::getline(file, line)) {
        lineno++;
        line = trim(line.substr(0, line.find('#')));
        if (line.empty()) continue;
        const size_t eq = line.find('=');
        if (eq == std::string::npos) CUDAERR("{}:{}: expected `key = value`, got `{}`", filename, lineno, line);
        const std::string key = trim(line.substr(0, eq));
        if (kv.count(key)) CUDAERR("{}:{}: `{}` is set twice", filename, lineno, key);
        kv[key] = trim(line.substr(eq+1));
    }

    // every key read below is erased, so whatever is left at the end is a typo or a stale key
    auto take = [&](const std::string& key, const char* fallback = nullptr) {
        auto it = kv.find(key);
        if (it == kv.end()) { if (fallback) return std::string(fallback); CUDAERR("{}: missing `{}`", filename, key); }
        std::string v = it->second; kv.erase(it); return v;
    };
    auto as_double = [&](const std::string& key, const std::string& v) {
        char* end = nullptr; const double d = std::strtod(v.c_str(), &end);    // strtod also takes hex floats, e.g. 0x1.99999ap+0
        if (v.empty() || *end != '\0') CUDAERR("{}: `{}` should be a number, got `{}`", filename, key, v);
        return d;
    };
    auto as_int = [&](const std::string& key, const std::string& v) {
        const double d = as_double(key, v);
        if (d != std::floor(d)) CUDAERR("{}: `{}` should be an integer, got `{}`", filename, key, v);
        return (int)d;
    };
    auto get_int = [&](const std::string& key, const char* fallback = nullptr) { return as_int(key, take(key, fallback)); };
    auto get_float = [&](const std::string& key, const char* fallback = nullptr) { return (float)as_double(key, take(key, fallback)); };
    auto in_range = [&](const char* key, double v, double lo, double hi) {
        if (v < lo || v > hi) CUDAERR("{}: `{}` = {} is outside [{}, {}]", filename, key, v, lo, hi);
    };

    EngineConfig c{};
    c.statespace_dim = get_int("statespace_dim");
    c.discrete_dim = get_int("discrete_dim");
    c.nonlin_dim = get_int("nonlin_dim");
    c.n_params = get_int("n_params");
    c.input_size = get_int("input_size");
    c.dc_instructions = get_int("dc_instructions");
    c.u_instructions = get_int("u_instructions");
    in_range("statespace_dim", c.statespace_dim, 1, MAX_STATESPACE_DIM);
    in_range("discrete_dim", c.discrete_dim, 1, MAX_DISCRETE_DIM);
    in_range("nonlin_dim", c.nonlin_dim, 0, MAX_NONLIN_DIM);
    in_range("n_params", c.n_params, 1, MAX_PARAMS);
    in_range("input_size", c.input_size, 0, 1<<20);
    in_range("dc_instructions", c.dc_instructions, 0, MAX_DC_INSTRUCTIONS);
    in_range("u_instructions", c.u_instructions, 0, MAX_U_INSTRUCTIONS);

    {
        std::stringstream list(take("nl_instructions", ""));
        std::string item;
        int n = 0;
        while (std::getline(list, item, ',')) {
            item = trim(item);
            if (item.empty()) continue;
            if (n >= MAX_NONLIN_DIM) CUDAERR("{}: `nl_instructions` has more than MAX_NONLIN_DIM = {} entries", filename, MAX_NONLIN_DIM);
            c.nl_instructions[n] = as_int("nl_instructions", item);
            in_range("nl_instructions", c.nl_instructions[n], 0, MAX_NL_INSTRUCTIONS);
            n++;
        }
        if (n != c.nonlin_dim) CUDAERR("{}: `nl_instructions` has {} entries but nonlin_dim is {}", filename, n, c.nonlin_dim);
    }

    c.n_cores = get_int("n_cores");
    c.n_rounds = get_int("n_rounds", "1");
    in_range("n_cores", c.n_cores, 128, 1<<20);
    if (c.n_cores % 128 != 0) CUDAERR("{}: `n_cores` = {} should be a multiple of 128 (cores per SM)", filename, c.n_cores);
    in_range("n_rounds", c.n_rounds, 1, 1<<20);
    {
        const std::string v = take("mode", "batch");
        if (v == "batch") c.serial = false;
        else if (v == "serial") c.serial = true;
        else CUDAERR("{}: `mode` should be batch/serial, got `{}`", filename, v);
        // the margin sweep swaps the true plant in for the observer's model, so their sizes have to match
        if (c.serial && c.statespace_dim != c.discrete_dim) CUDAERR("{}: mode = serial needs statespace_dim ({}) == discrete_dim ({})", filename, c.statespace_dim, c.discrete_dim);
    }

    c.time = get_int("time");
    c.ref_step_time = as_double("ref_step_time", take("ref_step_time"));
    c.ref_amplitude = get_float("ref_amplitude");
    if (!(c.ref_amplitude != 0.0f)) CUDAERR("{}: need ref_amplitude != 0", filename);    // steady-state error is relative to it
    in_range("time", c.time, 2, 1<<24);
    c.duration = get_float("duration", "1.0");
    if (!(c.duration > 0.0f)) CUDAERR("{}: need duration > 0", filename);
    in_range("ref_step_time", c.ref_step_time, 0.0, 1.0);

    c.fit_start = get_float("fit_start");
    c.fit_end = get_float("fit_end");
    c.steady_start = get_float("steady_start");
    in_range("fit_start", c.fit_start, 0.0, 1.0);
    in_range("fit_end", c.fit_end, 0.0, 1.0);
    in_range("steady_start", c.steady_start, 0.0, 1.0);
    if (!(c.fit_start < c.fit_end)) CUDAERR("{}: fit_start ({}) must be below fit_end ({})", filename, c.fit_start, c.fit_end);
    if ((int)(c.steady_start*c.time) >= c.time) CUDAERR("{}: steady_start leaves no samples to average", filename);

    c.bandwidth = get_float("bandwidth");
    c.damping = get_float("damping");
    c.overshoot = get_float("overshoot");
    c.steady_state_error = get_float("steady_state_error");
    c.crossover_coeff = get_float("crossover_coeff");
    c.target_gm = get_float("target_gm");
    c.target_pm = get_float("target_pm");
    c.target_b = get_float("target_b");
    if (!(c.overshoot > 0.0f)) CUDAERR("{}: need overshoot > 0", filename);    // overshoot divides err_os
    if (!(c.steady_state_error > 0.0f)) CUDAERR("{}: need steady_state_error > 0", filename);    // divides err_ss
    if (!(c.crossover_coeff > 0.0f)) CUDAERR("{}: need crossover_coeff > 0", filename);
    if (!(c.target_b > 0.0f && c.target_b < 1.0f)) CUDAERR("{}: need 0 < target_b < 1", filename);    // b_{P,C} is in [0,1], and target_b divides err_b
    c.eig_fail_err = get_float("eig_fail_err", "1.0e18");
    c.unstable_err = get_float("unstable_err", "1.0e6");

    c.freq = get_int("freq");
    c.freq_min = get_float("freq_min");
    in_range("freq", c.freq, 2, 1<<20);
    if (!(c.freq_min > 0.0f && c.freq_min < PI*c.time/c.duration)) CUDAERR("{}: need 0 < freq_min < pi*time/duration (Nyquist)", filename);

    {
        const std::string v = take("dump_diagnostics", "false");
        if (v == "true" || v == "1") c.dump_diagnostics = true;
        else if (v == "false" || v == "0") c.dump_diagnostics = false;
        else CUDAERR("{}: `dump_diagnostics` should be true/false, got `{}`", filename, v);
    }

    if (!kv.empty()) CUDAERR("{}: unknown key `{}`", filename, kv.begin()->first);

    c.derive();
    if (!(c.crossover > 0.0f && c.crossover < c.freq_max)) CUDAERR("{}: crossover {} rad/s (from bandwidth, damping, crossover_coeff) must be in (0, Nyquist = {})", filename, c.crossover, c.freq_max);
    return c;
}
