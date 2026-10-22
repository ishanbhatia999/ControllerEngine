#include <iostream>
#include <fstream>
#include <cstring>
#include <math.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <format>
#include <iomanip>
#include <complex>
#include <map>
#include <sstream>
#include <cmath>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#define EIGEN_NO_CUDA
#include <Eigen/Eigenvalues>

template<typename... Args>
void cudaerr(std::format_string<Args...> str, Args&&... args) {
    throw std::runtime_error(std::format(str, std::forward<Args>(args)...));
}
#define CUDAERR(...) cudaerr(__VA_ARGS__)

#define CUDARUN(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        CUDAERR("CUDA error {} at {}:{}", cudaGetErrorString(err), __FILE__, __LINE__); \
    } \
} while (0)

enum class ControllerType {
    Luenberger,
    Kalman,
    PID
};

constexpr float PI = 0x1.921fb6p+1f;

// MAX SIZE OF LIBRARY DEFINED HERE
constexpr int MAX_STATESPACE_DIM = 8;
constexpr int MAX_DISCRETE_DIM = 8;
constexpr int MAX_NONLIN_DIM = 4;
constexpr int MAX_TOTAL_DIM = MAX_STATESPACE_DIM + MAX_DISCRETE_DIM;
constexpr int MAX_NL_DIM = MAX_DISCRETE_DIM+1;    // xplant + yplant
constexpr int MAX_DC_INSTRUCTIONS = 4;
constexpr int MAX_U_INSTRUCTIONS = 4;
constexpr int MAX_NL_INSTRUCTIONS = 8;
constexpr int MAX_PARAMS = 34;    // upper limit of base 36

#include "config.cu"

__constant__ EngineParams ENGINE;

__constant__ float STATESPACE_A[MAX_STATESPACE_DIM*MAX_STATESPACE_DIM];
__constant__ float STATESPACE_A_I[MAX_STATESPACE_DIM*MAX_STATESPACE_DIM];
__constant__ float STATESPACE_B[MAX_STATESPACE_DIM];
__constant__ float STATESPACE_C[MAX_STATESPACE_DIM];
__constant__ float STATESPACE_D[1];
__constant__ float DISCRETE_A[MAX_DISCRETE_DIM*MAX_DISCRETE_DIM];
__constant__ float DISCRETE_A_I[MAX_DISCRETE_DIM*MAX_DISCRETE_DIM];
__constant__ float DISCRETE_BU[MAX_DISCRETE_DIM];
__constant__ float DISCRETE_BW[MAX_DISCRETE_DIM*MAX_NONLIN_DIM];
__constant__ float DISCRETE_C[MAX_DISCRETE_DIM];
__constant__ float BASE_PARAMS[MAX_PARAMS];

// where a decoded instruction's operand comes from. The packed digit encoding is: 0 -> the state
// sitting under the instruction's own column, 35 ('Z') -> the chain's running value, and 1..34 ->
// BASE_PARAMS[digit-1], which decode_instructions() resolves to a literal at load time.
enum : int { OPERAND_CONST = 0, OPERAND_STATE = 1, OPERAND_PREV = 2 };

struct Instruction {
    int   op;       // 0 SKIP, 1 ADD, 2 SUB, 3 MUL, 4 DIV, 5 SAT, 6 SGN
    int   col;      // column the row's nonzero sat under, indexes the chain's `source` array
    int   akind;    // OPERAND_* for the first operand
    int   bkind;    // OPERAND_* for the second operand (unused by SGN)
    float aval;     // resolved BASE_PARAMS value, meaningful only when the matching kind is OPERAND_CONST
    float bval;
};

// padded with SKIP rows; NL chain I starts at NL_OPS[I*MAX_NL_INSTRUCTIONS]
__constant__ Instruction DC_OPS[MAX_DC_INSTRUCTIONS];
__constant__ Instruction U_OPS[MAX_U_INSTRUCTIONS];
__constant__ Instruction NL_OPS[MAX_NONLIN_DIM*MAX_NL_INSTRUCTIONS];

// host copies keep the compact (unpadded, row-major) layout of the .bin files
float STATESPACE_A_HOST[MAX_STATESPACE_DIM*MAX_STATESPACE_DIM];
float STATESPACE_B_HOST[MAX_STATESPACE_DIM];
float STATESPACE_C_HOST[MAX_STATESPACE_DIM];
float STATESPACE_D_HOST[1];
float DISCRETE_A_HOST[MAX_DISCRETE_DIM*MAX_DISCRETE_DIM];
float DISCRETE_BU_HOST[MAX_DISCRETE_DIM];
float DISCRETE_C_HOST[MAX_DISCRETE_DIM];
float BASE_PARAMS_HOST[MAX_PARAMS];

__device__ float clampf(float X, float LB, float UB) {
    return fminf(fmaxf(X, LB), UB);
}

__device__ int signf(float X) {
    return (0<X) - (X<0);
}

template<int ROWS, int INNER, int COLS>
__device__ void matmul(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    float c = 0.0f;
    #pragma unroll
    for (int i=0; i<ROWS; i++) {
        #pragma unroll
        for (int j=0; j<COLS; j++) {
            c = 0.0f;
            #pragma unroll
            for (int k=0; k<INNER; k++) c = __fmaf_rn(A[i*INNER+k],B[k*COLS+j],c);
            C[i*COLS+j] = c;
        }
    }
}

template<int DIM>
__device__ __forceinline__ float vecmul(const float* __restrict__ A, const float* __restrict__ B, float acc) {
    #pragma unroll
    for (int k=0; k<DIM; k++) acc = __fmaf_rn(A[k], B[k], acc);
    return acc;
}

__device__ __forceinline__ void hlfast2sumf(float a, float b, float& hi, float& lo) {
    hi = __fadd_rn(a,b);
    lo = __fsub_rn(b,__fsub_rn(hi,a));
}

__device__ __forceinline__ void hl2sumf(float a, float b, float& hi, float& lo) {
    hi = __fadd_rn(a,b);
    const float b0 = __fsub_rn(hi,a);
    lo = __fadd_rn(__fsub_rn(a,__fsub_rn(hi,b0)), __fsub_rn(b,b0));
}

__device__ __forceinline__ void hlmulf(float a, float b, float& hi, float& lo) {
    hi = __fmul_rn(a,b);
    lo = __fmaf_rn(a,b,-hi);
}

__device__ __forceinline__ void hlprodf(float a, float b, float& hi, float& lo) {
    hi = __fmul_rn(a,b);
    lo = __fmaf_rn(a,b,-hi);
}

__device__ __forceinline__ void hladdf(float& HI, float& LO, float hi, float lo) {
    float sh, sl;
    hl2sumf(HI,hi,sh,sl);
    const float v = __fadd_rn(__fadd_rn(LO,lo),sl);
    hlfast2sumf(sh,v,HI,LO);
}

template<int ROWS, int COLS>
__device__ void matcpy(const float* __restrict__ REF, float* __restrict__ NEW) {
    #pragma unroll
    for (int i=0; i<ROWS; i++) {
        #pragma unroll
        for (int j=0; j<COLS; j++) {
            NEW[i*COLS+j] = REF[i*COLS+j];
        }
    }
}

template<int ROWS, int COLS>
__device__ void matadd(const float* __restrict__ A, const float* __restrict__ B, float* __restrict C) {
    #pragma unroll
    for (int i=0; i<ROWS; i++) {
        #pragma unroll
        for (int j=0; j<COLS; j++) {
            C[i*COLS+j] = __fadd_rn(A[i*COLS+j],B[i*COLS+j]);
        }
    }
}

template<int A_ROWS, int INNER, int B_ROWS>
__device__ __forceinline__ void ABTf(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    float c = 0.0f;
    #pragma unroll
    for (int i=0; i<A_ROWS; i++) {
        #pragma unroll
        for (int j=0; j<B_ROWS; j++) {
            c = 0.0f;
            #pragma unroll
            for (int k=0; k<INNER; k++) c = __fmaf_rn(A[i*INNER+k],B[j*INNER+k],c);
            C[i*B_ROWS+j] = c;
        }
    }
}

template<int A_ROWS, int DIM>
__device__ __forceinline__ void ABATf(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    float c = 0.0f, C0[A_ROWS*DIM];
    #pragma unroll
    for (int i=0; i<A_ROWS; i++) {
        #pragma unroll
        for (int j=0; j<DIM; j++) {
            c = 0.0f;
            #pragma unroll
            for (int k=0; k<DIM; k++) c = __fmaf_rn(A[i*DIM+k],B[k*DIM+j],c);
            C0[i*DIM+j] = c;
        }
    }

    #pragma unroll
    for (int i=0; i<A_ROWS; i++) {
        #pragma unroll
        for (int j=0; j<A_ROWS; j++) {
            c = 0.0f;
            #pragma unroll
            for (int k=0; k<DIM; k++) c = __fmaf_rn(C0[i*DIM+k],A[j*DIM+k],c);
            C[i*A_ROWS+j] = c;
        }
    }
}


// C = A + B over LEN entries
template<int LEN>
__device__ void vecadd(const float* A, const float* B, float* C) {
    #pragma unroll
    for (int i=0; i<LEN; i++) C[i] = A[i] + B[i];
}
template<int LEN>
__device__ void vecadd(const float* A, const float* B, double* C) {
    #pragma unroll
    for (int i=0; i<LEN; i++) C[i] = static_cast<double>(A[i]) + static_cast<double>(B[i]);
}

// C = A - B over LEN entries
template<int LEN>
__device__ void vecsub(const float* A, const float* B, float* C) {
    #pragma unroll
    for (int i=0; i<LEN; i++) C[i] = A[i] - B[i];
}

// C = A + B + D over LEN entries
template<int LEN>
__device__ void vecadd3(const float* A, const float* B, const float* D, float* C) {
    #pragma unroll
    for (int i=0; i<LEN; i++) C[i] = A[i] + B[i] + D[i];
}
template<int LEN>
__device__ void vecadd3(const float* A, const float* B, const float* D, double* C) {
    #pragma unroll
    for (int i=0; i<LEN; i++) C[i] = static_cast<double>(A[i]) + static_cast<double>(B[i]) + static_cast<double>(D[i]);
}

// C = A - B - D over LEN entries
template<int LEN>
__device__ void vecsub3(const float* A, const float* B, const float* D, float* C) {
    #pragma unroll
    for (int i=0; i<LEN; i++) C[i] = A[i] - B[i] - D[i];
}


template<int DIM, int COUNT, bool HAS_SEED>
__device__ __forceinline__ float interpreter(const Instruction* __restrict__ info, const float* __restrict__ source, float seed = 0.0f) {
    float result = 0.0f;
    bool started = false;

    #pragma unroll
    for (int r = 0; r < COUNT; r++) {
        const Instruction ins = info[r];
        if (ins.op == 0) continue;

        float posState = 0.0f;
        #pragma unroll
        for (int c = 0; c < DIM; c++) if (c == ins.col) posState = source[c];

        const float prevValue = started ? result : ((HAS_SEED && !isnan(seed)) ? seed : posState);

        const float a = ins.akind == OPERAND_CONST ? ins.aval : (ins.akind == OPERAND_STATE ? posState : prevValue);
        const float b = ins.bkind == OPERAND_CONST ? ins.bval : (ins.bkind == OPERAND_STATE ? posState : prevValue);

        switch (ins.op) {
            case 1: result = a + b; break;
            case 2: result = a - b; break;
            case 3: result = a * b; break;
            case 4: result = a / b; break;
            case 5: result = clampf(prevValue, a, b); break;
            case 6: result = (float)signf(a); break;
        }
        started = true;
    }

    return result;
}

// a file can hold `slices` back-to-back copies of nbytes (serial mode stacks one plant per round); this reads copy `slice`
__host__ void readbin(size_t nbytes, const std::string& filename, void* host_dst, int slice = 0, int slices = 1) {
    std::ifstream file(filename, std::ios::binary); if (!file) CUDAERR("Cannot open {}", filename);
    file.seekg(0,std::ios::end);
    const size_t expected = nbytes*slices;
    std::streamoff actual = file.tellg(); if (actual != static_cast<std::streamoff>(expected)) CUDAERR("{} is {} bytes, expected {} ({} x {})", filename, actual, expected, slices, nbytes);
    file.seekg(static_cast<std::streamoff>(nbytes*slice),std::ios::beg);
    file.read(reinterpret_cast<char*>(host_dst), nbytes); if (!file) CUDAERR("Cannot read {} (intended {}-byte allocation)", filename, nbytes);
    file.close();
}

// reads a row-major rows x cols matrix and scatters it into a zeroed __constant__ array whose rows are
// `stride` floats apart (the MAX_* padded layout). host_mirror, if given, gets the compact rows x cols copy.
template<size_t N>
__host__ void upload_padded(const float (&symbol)[N], const std::string& filename, int rows, int cols, int stride, float* host_mirror = nullptr, int slice = 0, int slices = 1) {
    if (cols > stride || static_cast<size_t>((rows-1)*stride + cols) > N) CUDAERR("{} is {}x{}, which does not fit its padded {}-float constant array", filename, rows, cols, N);
    std::vector<float> compact(static_cast<size_t>(rows)*cols);
    readbin(compact.size()*sizeof(float), filename, compact.data(), slice, slices);
    if (host_mirror) std::memcpy(host_mirror, compact.data(), compact.size()*sizeof(float));    // copies to host
    std::vector<float> padded(N, 0.0f);
    for (int i=0; i<rows; i++) for (int j=0; j<cols; j++) padded[i*stride+j] = compact[i*cols+j];
    CUDARUN(cudaMemcpyToSymbol(symbol, padded.data(), N*sizeof(float)));    // copies to device
}

__host__ void decode_instructions(const int* info, int dim, int count, Instruction* out) {
    auto kind = [](int digit) { return digit == 0 ? OPERAND_STATE : (digit == 35 ? OPERAND_PREV : OPERAND_CONST); };
    auto val  = [](int digit) {
        if (digit == 0 || digit == 35) return 0.0f;
        if (digit-1 >= CFG.n_params) CUDAERR("instruction digit {} needs BASE_PARAMS[{}] but n_params is {}", digit, digit-1, CFG.n_params);
        return BASE_PARAMS_HOST[digit-1];
    };

    for (int r=0; r<count; r++) {
        const int* row = &info[r*dim];
        int packed = 0, col = 0;
        for (int c=0; c<dim; c++) if (row[c] != 0) { packed = row[c]; col = c; }

        out[r] = Instruction{0, 0, OPERAND_CONST, OPERAND_CONST, 0.0f, 0.0f};
        if (packed == 0) continue;

        const int i1 = (packed / 36) % 36;
        const int i2 = packed % 36;
        out[r] = Instruction{packed / 1296, col, kind(i1), kind(i2), val(i1), val(i2)};
    }
}

#include "model_functions.cuh"

__host__ void upload_plant(int round) {
    const int ds = CFG.discrete_dim;
    const int slices = CFG.serial ? CFG.n_rounds : 1;
    const int slice = CFG.serial ? round : 0;

    // DISCRETE A, BU, BW, C MATRICES
    upload_padded(DISCRETE_A,   "DISCRETE_A.bin",   ds, ds, MAX_DISCRETE_DIM, DISCRETE_A_HOST, slice, slices);
    upload_padded(DISCRETE_BU,  "DISCRETE_BU.bin",  ds, 1,  1, DISCRETE_BU_HOST, slice, slices);
    upload_padded(DISCRETE_BW,  "DISCRETE_BW.bin",  ds, CFG.nonlin_dim, MAX_NONLIN_DIM, nullptr, slice, slices);
    upload_padded(DISCRETE_C,   "DISCRETE_C.bin",   1,  ds, MAX_DISCRETE_DIM, DISCRETE_C_HOST, slice, slices);
    upload_padded(DISCRETE_A_I, "DISCRETE_A_I.bin", ds, ds, MAX_DISCRETE_DIM, nullptr, slice, slices);

    // BASE_PARAMS
    upload_padded(BASE_PARAMS, "BASE_PARAMS.bin", 1, CFG.n_params, MAX_PARAMS, BASE_PARAMS_HOST, slice, slices);

    // INSTRUCTION SETS
    std::vector<int> dc_packed(static_cast<size_t>(CFG.dc_dim) * CFG.dc_instructions);
    std::vector<int> u_packed(static_cast<size_t>(CFG.u_dim) * CFG.u_instructions);
    std::vector<int> nl_packed(static_cast<size_t>(CFG.nl_dim) * CFG.nl_instructions_total);
    readbin(dc_packed.size()*sizeof(int), "DC_INFO.bin", dc_packed.data());
    readbin(u_packed.size()*sizeof(int),  "U_INFO.bin",  u_packed.data());
    readbin(nl_packed.size()*sizeof(int), "NL_INFO.bin", nl_packed.data());

    // unused slots stay SKIP (op 0), which interpreter() passes over
    const Instruction skip{0, 0, OPERAND_CONST, OPERAND_CONST, 0.0f, 0.0f};
    Instruction dc_ops[MAX_DC_INSTRUCTIONS];
    Instruction u_ops[MAX_U_INSTRUCTIONS];
    Instruction nl_ops[MAX_NONLIN_DIM*MAX_NL_INSTRUCTIONS];
    std::fill(std::begin(dc_ops), std::end(dc_ops), skip);
    std::fill(std::begin(u_ops),  std::end(u_ops),  skip);
    std::fill(std::begin(nl_ops), std::end(nl_ops), skip);
    decode_instructions(dc_packed.data(), CFG.dc_dim, CFG.dc_instructions, dc_ops);
    decode_instructions(u_packed.data(),  CFG.u_dim,  CFG.u_instructions,  u_ops);
    for (int i=0, row=0; i<CFG.nonlin_dim; row += CFG.nl_instructions[i], i++) {
        Instruction* chain = &nl_ops[i*MAX_NL_INSTRUCTIONS];
        decode_instructions(&nl_packed[static_cast<size_t>(row)*CFG.nl_dim], CFG.nl_dim, CFG.nl_instructions[i], chain);
        for (int r=0; r<CFG.nl_instructions[i]; r++) if (chain[r].col == CFG.discrete_dim) chain[r].col = MAX_DISCRETE_DIM;
    }

    CUDARUN(cudaMemcpyToSymbol(DC_OPS, dc_ops, sizeof(dc_ops)));
    CUDARUN(cudaMemcpyToSymbol(U_OPS,  u_ops,  sizeof(u_ops)));
    CUDARUN(cudaMemcpyToSymbol(NL_OPS, nl_ops, sizeof(nl_ops)));
}

__host__ void init64KB() {
    CFG = load_config("ControllerEngine.config");
    const EngineParams engine = make_engine_params(CFG);
    CUDARUN(cudaMemcpyToSymbol(ENGINE, &engine, sizeof(engine)));

    const int ss = CFG.statespace_dim;

    // STATESPACE A B C D MATRICES
    upload_padded(STATESPACE_A,   "STATESPACE_A.bin",   ss, ss, MAX_STATESPACE_DIM, STATESPACE_A_HOST);
    upload_padded(STATESPACE_A_I, "STATESPACE_A_I.bin", ss, ss, MAX_STATESPACE_DIM);
    upload_padded(STATESPACE_B,   "STATESPACE_B.bin",   ss, 1,  1, STATESPACE_B_HOST);
    upload_padded(STATESPACE_C,   "STATESPACE_C.bin",   1,  ss, MAX_STATESPACE_DIM, STATESPACE_C_HOST);
    upload_padded(STATESPACE_D,   "STATESPACE_D.bin",   1,  1,  1, STATESPACE_D_HOST);

    upload_plant(0);
}

__host__ void orchestrate_sims() {
    std::ofstream plog("params.log");
    plog << std::scientific << std::setprecision(9);
    for (int i=0; i<CFG.n_params; i++) { plog << "BASE_PARAMS_HOST[" << i << "] = " << BASE_PARAMS_HOST[i] << "\n"; }
    plog << "ref amplitude = " << CFG.ref_amplitude << "\n";

    int threadsPerBlock = 256;
    int blocks = (CFG.n_threads + threadsPerBlock-1)/threadsPerBlock;

    // SET UP REFERENCE INPUT
    float* host_ptr = nullptr;
    CUDARUN(cudaMallocHost(reinterpret_cast<void**>(&host_ptr), CFG.time*sizeof(float)));
    for (int t=0; t<CFG.time; t++) host_ptr[t] = (t<CFG.ref_step_time*CFG.time ? 0.0f : CFG.ref_amplitude);    // add support for multiple inputs later

    float* ref_ptr = nullptr;
    CUDARUN(cudaMalloc(reinterpret_cast<void**>(&ref_ptr), CFG.time*sizeof(float)));
    CUDARUN(cudaMemcpy(ref_ptr, host_ptr, CFG.time*sizeof(float), cudaMemcpyHostToDevice));
    CUDARUN(cudaFreeHost(host_ptr));

    // SET UP Z MATRIX (log-spaced frequency sweep, z = e^{j*fw*TS})
    std::complex<float>* z_ptr = nullptr;
    CUDARUN(cudaMallocHost(reinterpret_cast<void**>(&z_ptr), CFG.freq*sizeof(std::complex<float>)));
    float increment = (logf(CFG.freq_max)-logf(CFG.freq_min))/(CFG.freq-1);
    for (int i=0; i<CFG.freq; i++) {
        float theta = expf(logf(CFG.freq_min)+increment*i)*CFG.ts;
        z_ptr[i] = std::complex<float>(cosf(theta), sinf(theta));
    }

    // DEVICE BUFFERS PER ROUND W/ N_THREADS CANDIDATES
    float* cuda_ptr = nullptr;   CUDARUN(cudaMalloc(reinterpret_cast<void**>(&cuda_ptr),   CFG.cuda_mx_size*sizeof(float)));
    float* yplant_ptr = nullptr; CUDARUN(cudaMalloc(reinterpret_cast<void**>(&yplant_ptr), CFG.yplant_size*sizeof(float)));
    float* eigen_ptr = nullptr;  CUDARUN(cudaMalloc(reinterpret_cast<void**>(&eigen_ptr),  CFG.eigen_mx_size*sizeof(float)));
    float* lina_ptr = nullptr;   CUDARUN(cudaMalloc(reinterpret_cast<void**>(&lina_ptr),   CFG.lina_mx_size*sizeof(float)));
    float* ata_ptr = nullptr;    CUDARUN(cudaMalloc(reinterpret_cast<void**>(&ata_ptr),    CFG.ata_mx_size*sizeof(float)));
    float* atb_ptr = nullptr;    CUDARUN(cudaMalloc(reinterpret_cast<void**>(&atb_ptr),    CFG.atb_mx_size*sizeof(float)));
    float* info_ptr = nullptr;   CUDARUN(cudaMalloc(reinterpret_cast<void**>(&info_ptr),   CFG.info_mx_size*sizeof(float)));

    // HOST BUFFERS PER ROUND W/ N_THREADS CANDIDATES
    float* cuda_mx_round = nullptr; CUDARUN(cudaMallocHost(reinterpret_cast<void**>(&cuda_mx_round), CFG.cuda_mx_size*sizeof(float)));
    float* yplant_host = nullptr;   if (CFG.dump_diagnostics) CUDARUN(cudaMallocHost(reinterpret_cast<void**>(&yplant_host), CFG.yplant_size*sizeof(float)));
    float* lina_host = nullptr;     CUDARUN(cudaMallocHost(reinterpret_cast<void**>(&lina_host),     CFG.lina_mx_size*sizeof(float)));
    float* eigen_host = nullptr;    CUDARUN(cudaMallocHost(reinterpret_cast<void**>(&eigen_host),    CFG.eigen_mx_size*sizeof(float)));
    float* ata_host = nullptr;      CUDARUN(cudaMallocHost(reinterpret_cast<void**>(&ata_host),      CFG.ata_mx_size*sizeof(float)));
    float* atb_host = nullptr;      CUDARUN(cudaMallocHost(reinterpret_cast<void**>(&atb_host),      CFG.atb_mx_size*sizeof(float)));
    float* info_mx_round = nullptr; CUDARUN(cudaMallocHost(reinterpret_cast<void**>(&info_mx_round), CFG.info_mx_size*sizeof(float)));

    // INPUT GAINS MX
    std::ifstream cudamx("CUDA_MX.bin", std::ios::binary);
    if (!cudamx) { CUDAERR("Cannot open CUDA_MX.bin"); }
    cudamx.seekg(0, std::ios::end);
    const std::streamoff cudamx_expected = static_cast<std::streamoff>(CFG.total_candidates*CFG.cuda_mx_width*sizeof(float));
    if (cudamx.tellg() != cudamx_expected) { CUDAERR("CUDA_MX.bin is {} bytes, expected {} ({} rounds x {} candidates)", static_cast<long long>(cudamx.tellg()), static_cast<long long>(cudamx_expected), CFG.n_rounds, CFG.n_threads); }
    cudamx.seekg(0, std::ios::beg);

    // INPUT PARAMS + OUTPUT SCORE MX
    std::ifstream infomx("INFO_MX.bin", std::ios::binary);
    if (!infomx) { CUDAERR("Cannot open INFO_MX.bin"); }
    infomx.seekg(0, std::ios::end);
    const std::streamoff infomx_expected = static_cast<std::streamoff>(CFG.total_candidates*CFG.info_mx_width*sizeof(float));
    if (infomx.tellg() != infomx_expected) { CUDAERR("INFO_MX.bin is {} bytes, expected {} ({} rounds x {} candidates)", static_cast<long long>(infomx.tellg()), static_cast<long long>(infomx_expected), CFG.n_rounds, CFG.n_threads); }
    infomx.seekg(0, std::ios::beg);

    std::ofstream infoout("INFO_MX_OUT.bin", std::ios::binary);
    if (!infoout) { CUDAERR("Cannot open INFO_MX_OUT.bin"); }

    for (int round=0; round<CFG.n_rounds; round++) {
        // SET UP THIS ROUND'S PLANT (serial mode only, round 0 is already loaded by init64KB)
        if (CFG.serial && round > 0) upload_plant(round);

        // SET UP INPUT GAINS
        cudamx.read(reinterpret_cast<char*>(cuda_mx_round), CFG.cuda_mx_size*sizeof(float));
        if (!cudamx) { CUDAERR("Cannot read CUDA_MX.bin round {} (intended {}-byte allocation)", round, CFG.cuda_mx_size*sizeof(float)); }
        CUDARUN(cudaMemcpy(cuda_ptr, cuda_mx_round, CFG.cuda_mx_size*sizeof(float), cudaMemcpyHostToDevice));

        // RUN SIMULATION
        CUDARUN(cudaMemset(yplant_ptr, 0, CFG.yplant_size*sizeof(float)));
        discrete_solve<ControllerType::Luenberger><<<blocks, threadsPerBlock>>>(cuda_ptr, ref_ptr, yplant_ptr);
        CUDARUN(cudaGetLastError());
        CUDARUN(cudaDeviceSynchronize());

        // PULL THE SIMULATION OUTPUT BACK OUT
        if (CFG.dump_diagnostics) {
            CUDARUN(cudaMemcpy(yplant_host, yplant_ptr, CFG.yplant_size*sizeof(float), cudaMemcpyDeviceToHost));
            std::ofstream yout("YPLANT_OUT.bin", std::ios::binary);
            if (!yout) { CUDAERR("Cannot open YPLANT_OUT.bin"); }
            yout.write(reinterpret_cast<char*>(yplant_host), CFG.yplant_size*sizeof(float));
            yout.close();
        }

        // SET UP MATRICES FOR EIGENVALUE SOLVE
        CUDARUN(cudaMemset(lina_ptr, 0, CFG.lina_mx_size*sizeof(float)));
        eigensetup<<<blocks, threadsPerBlock>>>(cuda_ptr, lina_ptr);
        CUDARUN(cudaGetLastError());
        CUDARUN(cudaDeviceSynchronize());

        // PULL LINA MATRIX BACK OUT (both CPU and GPU)
        CUDARUN(cudaMemcpy(lina_host, lina_ptr, CFG.lina_mx_size*sizeof(float), cudaMemcpyDeviceToHost));
        if (CFG.dump_diagnostics) {
            std::ofstream linaout("LINA_MX_OUT.bin", std::ios::binary);
            if (!linaout) { CUDAERR("Cannot open LINA_MX_OUT.bin"); }
            linaout.write(reinterpret_cast<char*>(lina_host), CFG.lina_mx_size*sizeof(float));
            linaout.close();
        }

        // COMPUTE EIGENVALUES ON LINA (CPU)
        eigensolve(lina_host, eigen_host);

        // TRANSFER EIGENVALUES BACK TO GPU FOR MODALSETUP
        CUDARUN(cudaMemset(eigen_ptr, 0, CFG.eigen_mx_size*sizeof(float)));
        CUDARUN(cudaMemcpy(eigen_ptr, eigen_host, CFG.eigen_mx_size*sizeof(float), cudaMemcpyHostToDevice));
        if (CFG.dump_diagnostics) {
            std::ofstream eigenout("EIGEN_MX_OUT.bin", std::ios::binary);
            if (!eigenout) { CUDAERR("Cannot open EIGEN_MX_OUT.bin"); }
            eigenout.write(reinterpret_cast<char*>(eigen_host), CFG.eigen_mx_size*sizeof(float));
            eigenout.close();
        }

        // SET UP MATRICES FOR MODAL FIT
        CUDARUN(cudaMemset(ata_ptr, 0, CFG.ata_mx_size*sizeof(float)));
        CUDARUN(cudaMemset(atb_ptr, 0, CFG.atb_mx_size*sizeof(float)));

        // SET UP INFO MATRIX (GPU) -- this round's slice of the pre-flagged input
        infomx.read(reinterpret_cast<char*>(info_mx_round), CFG.info_mx_size*sizeof(float));
        if (!infomx) { CUDAERR("Cannot read INFO_MX.bin round {} (intended {}-byte allocation)", round, CFG.info_mx_size*sizeof(float)); }
        CUDARUN(cudaMemcpy(info_ptr, info_mx_round, CFG.info_mx_size*sizeof(float), cudaMemcpyHostToDevice));

        modalsetup<<<blocks, threadsPerBlock>>>(yplant_ptr, eigen_ptr, ata_ptr, atb_ptr, info_ptr);
        CUDARUN(cudaGetLastError());
        CUDARUN(cudaDeviceSynchronize());

        // PULL THE MATRICES FOR MODAL FIT BACK OUT (both CPU and GPU)
        CUDARUN(cudaMemcpy(ata_host, ata_ptr, CFG.ata_mx_size*sizeof(float), cudaMemcpyDeviceToHost));
        if (CFG.dump_diagnostics) {
            std::ofstream ataout("ATA_MX_OUT.bin", std::ios::binary);
            if (!ataout) { CUDAERR("Cannot open ATA_MX_OUT.bin"); }
            ataout.write(reinterpret_cast<char*>(ata_host), CFG.ata_mx_size*sizeof(float));
            ataout.close();
        }

        CUDARUN(cudaMemcpy(atb_host, atb_ptr, CFG.atb_mx_size*sizeof(float), cudaMemcpyDeviceToHost));
        if (CFG.dump_diagnostics) {
            std::ofstream atbout("ATB_MX_OUT.bin", std::ios::binary);
            if (!atbout) { CUDAERR("Cannot open ATB_MX_OUT.bin"); }
            atbout.write(reinterpret_cast<char*>(atb_host), CFG.atb_mx_size*sizeof(float));
            atbout.close();
        }

        // PULL INFO MATRIX BACK OUT
        CUDARUN(cudaMemcpy(info_mx_round, info_ptr, CFG.info_mx_size*sizeof(float), cudaMemcpyDeviceToHost));

        // SOLVE THE MODAL FIT AND SCORE BEHAVIOR
        modalsolve(ata_host, atb_host, eigen_host, info_mx_round);

        // SCORE STABILITY MARGINS --> TOTAL ERROR
        allmargin(lina_host, cuda_mx_round, z_ptr, info_mx_round);

        infoout.write(reinterpret_cast<char*>(info_mx_round), CFG.info_mx_size*sizeof(float));
    }

    cudamx.close();
    infomx.close();
    infoout.close();

    CUDARUN(cudaFreeHost(cuda_mx_round));
    if (CFG.dump_diagnostics) CUDARUN(cudaFreeHost(yplant_host));
    CUDARUN(cudaFreeHost(lina_host));
    CUDARUN(cudaFreeHost(eigen_host));
    CUDARUN(cudaFreeHost(ata_host));
    CUDARUN(cudaFreeHost(atb_host));
    CUDARUN(cudaFreeHost(info_mx_round));
    CUDARUN(cudaFreeHost(z_ptr));

    CUDARUN(cudaFree(ref_ptr));
    CUDARUN(cudaFree(cuda_ptr));
    CUDARUN(cudaFree(yplant_ptr));
    CUDARUN(cudaFree(info_ptr));
    CUDARUN(cudaFree(eigen_ptr));
    CUDARUN(cudaFree(lina_ptr));
    CUDARUN(cudaFree(ata_ptr));
    CUDARUN(cudaFree(atb_ptr));
}


extern "C" __declspec(dllexport) int RunControllerEngine() {
    try{
        init64KB();
        orchestrate_sims();
        return 0;
    } catch (const std::exception& e) {
        std::ofstream elog("error.log", std::ios::app);
        elog << "ControllerEngine error: " << e.what() << std::endl;
        return 1;
    }
}

