# ControllerEngine
ControllerEngine is a GPU-accelerated controller tuning library compatible with nonlinear plants able to be decomposed in the form x' = A\*x + B_u\*u + B_w\*w, where B_w is not a true disturbance matrix but is instead used to update (per-timestep) nonlinear terms in the system's governing equations.

The shipped build configures maximum plant parameters as:
- 8 plant and controller states
- 4 nonlinear terms (width of B_w)
- 4 nonlinear controller instructions (i.e. post-controller nonlinear steps)
- 4 nonlinear output instructions (i.e. steps taken to compute the plant output)
- 8 nonlinear plant instructions (i.e. nonlinearities in the plant itself)

These are shipped as detailed to be compatible with most SISO controller projects without excessive compute overhead. If your project requires different maximums, feel free to edit them in main.cu and rebuild with the following command:

```nvcc -shared -cudart static -std=c++20 -o ControllerEngine.dll src\main.cu -I lib\eigen -lcublas -Xcompiler /openmp -Xcompiler /bigobj -Xcompiler /O2```

## v1
Supported performance specs: overshoot, steady-state tracking error, damping ratio, bandwidth frequency (computed from the step response via participation factor analysis of eigenvalues of linearized state-space system)

Supported stability specs: gain margin (I recommend 9-12dB for global search, 6dB for Monte Carlo), phase margin (I recommend 55-65deg for global search, 45deg for Monte Carlo)

Supported robustness specs: Vinnicombe margin (I recommend 0.2)