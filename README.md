# ControllerEngine-v1

ControllerEngine is a GPU-accelerated controller tuning library compatible with nonlinear plants able to be decomposed in the form:

\[
x' = A*x + B_u*u + B_w*w
\]

where \(B_w\) is not a true disturbance matrix but is instead used to update (per-timestep) nonlinear terms in the system's governing equations.

The shipped build configures maximum plant parameters as:

- 8 plant and controller states
- 4 nonlinear terms (width of \(B_w\))
- 4 nonlinear controller instructions (i.e. post-controller nonlinear steps)
- 4 nonlinear output instructions (i.e. steps taken to compute the plant output)
- 8 nonlinear plant instructions (i.e. nonlinearities in the plant itself)

These are shipped as detailed to be compatible with most SISO controller projects without excessive compute overhead. If your project requires different maximums, feel free to edit them in `main.cu` and rebuild with the following command:

```bash
nvcc -shared -cudart static -std=c++20 -o ControllerEngine.dll src\main.cu -I lib\eigen -lcublas -Xcompiler /openmp -Xcompiler /bigobj -Xcompiler /O2
```

## v1

**Supported performance requirements:**

- Overshoot
- Steady-state tracking error
- Damping ratio
- Bandwidth frequency (computed from the step response via participation factor analysis of eigenvalues of linearized state-space system)

**Supported stability requirements:**

- Gain margin (I recommend 9-12dB for global search, 6dB for Monte Carlo)
- Phase margin (I recommend 55-65deg for global search, 45deg for Monte Carlo)

**Supported robustness requirements:**

- Vinnicombe margin (I recommend 0.2)

## Usage

### .BIN Files

**NONLINEAR PLANT:** `x' = A*x + B_u*u + B_w*w`

- `DISCRETE_A.BIN`, `DISCRETE_A_I.BIN` (A-I^n)
- `DISCRETE_BU.BIN`
- `DISCRETE_BW.BIN`

**LINEARIZED PLANT:** state space A, B, C, D

- `DISCRETE_C.BIN`

**OBSERVER MODEL:** state space A, B, C, D (note: you can match statespace A/B/C/D to that of the linearized plant's)

- `STATESPACE_A.BIN`, `STATESPACE_A_I.BIN` (A-I^n)
- `STATESPACE_B.BIN`
- `STATESPACE_C.BIN`
- `STATESPACE_D.BIN`

`BASE_PARAMS.BIN`: Constants used in the plant and observer models

`CUDA_MX.BIN`: Controller gains (K,L,N)

`INFO_MX.BIN`: Input/output

`DC_INFO.BIN`, `U_INFO.BIN`, `NL_INFO.BIN`: Nonlinear instructions provided in base-36, where each row encodes an instruction. The row's size is determined by the matrix the interpreter acts on, so:

- `DC_INFO`: computes the plant output from the plant states ---> discrete dimension
- `U_INFO`: computes the controller output from the controller states --> statespace dimension
- `NL_INFO`: computes nonlinearities within the plant --> discrete dimension

Within a row, an entry is read to execute an instruction, within which each digit encodes information that the interpreter reads:

- First digit: operation
- Second digit: value 1
- Third digit: value 2

Therefore, each entry:

\[
\text{entry} = \text{digit1}*36^2 + \text{digit2}*36 + \text{digit3}
\]

**Operations:**

- 1 (value1+value2)
- 2 (value1-value2)
- 3 (value1*value2)
- 4 (value1/value2)
- 5 (clamp the value computed prior to the pending instruction to [value1,value2])
- 6 (sign of value1, value2 ignored)

The entry's position within the row matters as well:

- `0` encodes the value of the state whose position in the plant model matches the entry's position within the row
- `Z` encodes the value computed prior to the pending instruction

Values 1-Y are reserved for constants supplied by `BASE_PARAMS.BIN`. I suggest you add constants like 1 and -1 to `BASE_PARAMS.BIN` to make your life easier when writing instruction sets.

**Instruction matrix dimensions:**

| File | Description | Dimensions |
|---|---|---|
| `DC_INFO.BIN` | plant output y | `dc_instructions x discrete_dim` |
| `U_INFO.BIN` | controller output | `u_instructions x statespace_dim` |
| `NL_INFO.BIN` | nonlinear terms w | `sum(nl_instructions) x (discrete_dim+1)` |

- `NL_INFO.BIN` encodes instruction sets for ALL nonlinearities in the plant; evaluated in order (nonlinearity 0 ... N), set lengths can be configured (see below).
- Additional dimension is for the plant output in case you need that along with the plant states.

### .config File Layout

```text
statespace_dim
discrete_dim
nonlin_dim (how many nonlinearities you have in the plant model, used with NL_INFO.BIN)

n_params (sizes BASE_PARAMS)
input_size (# of gains provided in CUDA_MX.BIN)

dc_instructions
u_instructions
nl_instructions (provide in the format nl1,nl2,nl3,nl4,...,nlN)


n_cores (default to the # of cores in your GPU, use a smaller number for Monte Carlo)
n_rounds (default to 1, only use larger numbers for Monte Carlo)

mode (serial, batch)


time (number of timesteps in your simulation)
duration
ref_step_time (when the step input starts)
ref_amplitude (amplitude of the step input)


fit_start (when you want the modal fit to start)
fit_end (when you want the modal fit to end)

steady_start (when do you think the system will realistically settle? this will define when the steady state value begins getting computed)

bandwidth
damping
overshoot (provide as a decimal, e.g. 10% overshoot is 0.10)
steady_state_error (provide as a decimal, e.g. 5% error is 0.05)

crossover_coeff (b-margin plant scaling at coeff * 2nd-order gain crossover of the spec)
target_gm
target_pm
target_b (Vinnicombe)

eig_fail_err (if eigensolve.cu fails, what should the penalty be? if you are unsure, define as 1e18)
unstable_err (if the system is determined to be unstable anywhere in the solve, what should the penalty be? if you are unsure, define as 1e6)


freq (# frequencies in the logspace for the frequency response)
freq_min (when the frequency response should start. maximum is the Nyquist frequency)


dump_diagnostics = false
```