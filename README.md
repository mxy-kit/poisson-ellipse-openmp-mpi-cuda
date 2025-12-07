# 2D Poisson Equation on an Elliptic Domain (MPI + OpenMP + CUDA)

This project implements and studies parallel algorithms for solving the
2-dimensional Poisson equation on an elliptic domain using the **fictitious
domain method** and a **preconditioned Conjugate Gradient (PCG)** solver.

Several implementations are compared:

- **Sequential CPU version**
- **OpenMP** (shared–memory parallelism)
- **MPI** (distributed–memory, 2D domain decomposition)
- **MPI + OpenMP** (hybrid CPU version)
- **MPI + CUDA** (hybrid CPU–GPU version, one GPU per MPI rank)

The code and experiments were written for the *Polus* cluster (MSU), but can be
adapted to any Linux cluster with MPI and CUDA support.

---

## 1. Problem statement

We solve the Poisson equation

\[
 -\Delta u(x,y) = f(x,y)
\]

inside the domain

\[
D = \{(x,y): x^2 + 4y^2 < 1\},
\]

embedded into the rectangular box

\[
\Omega = [A_1,B_1]\times[A_2,B_2] = [-1,1]\times[-0.6,0.6],
\]

with homogeneous Dirichlet boundary conditions

\[
 u(x,y) = 0 \quad \text{on } \partial D.
\]

An analytical solution

\[
 u(x,y) = \frac{1}{10}\bigl(1 - x^2 - 4y^2\bigr)
\]

is used to control the accuracy of the numerical solution.

### Fictitious domain method

The idea is to extend the PDE to the whole rectangle \(\Omega\) and use a
piecewise constant coefficient \(k(x,y)\) in

\[
 -\nabla\!\cdot\!\bigl(k(x,y)\,\nabla u(x,y)\bigr) = f(x,y):
\]

- inside the ellipse \(D\): \(k(x,y)=1\);
- outside \(D\): \(k(x,y)=1/\varepsilon\), where  
  \(\varepsilon = \max(h_x,h_y)^2\) and \(h_x,h_y\) are grid steps.

The rectangle is discretised by a uniform grid of size \(M\times N\).  
A finite–difference scheme leads to a large sparse linear system which is
solved by PCG with a diagonal preconditioner.

---

## 2. Implemented methods

All versions use the same numerical scheme, differ only in the way parallelism
is organised.

### Sequential CPU

- Builds the full grid on \(\Omega\).
- Computes fictitious domain coefficients \(a_{ij}, b_{ij}\) and the right-hand side.
- Runs PCG with a diagonal preconditioner until
  \(\|w^{k+1}-w^k\|_2 < 10^{-6}\).

Used as the reference to compute speedup \(S = T_{\text{seq}}/T\).

### OpenMP

- Shared–memory parallelisation of the main loops using
  `#pragma omp parallel for` (and `reduction(+:...)` where needed).
- Parallelised parts:
  - computation of \(a_{ij}, b_{ij}, f_{ij}\),
  - application of the operator \(A\),
  - application of \(D^{-1}\),
  - updates of vectors \(w,r,z,p\) and norms.

### MPI (2D domain decomposition)

- Rectangle \((1..M-1)\times(1..N-1)\) is split into a grid of subdomains
  \(P_x\times P_y\) (`choose_process_grid`).
- Each MPI process owns a rectangular block of internal nodes plus halo layers.
- Halo exchange is done with `MPI_Sendrecv` before applying \(A\).
- Global inner products and norms are computed via `MPI_Allreduce`.
- The PCG algorithm is implemented in `gradient_solver_mpi`.

### MPI + OpenMP (CPU hybrid)

- Same 2D domain decomposition and MPI communication as in the pure MPI code.
- Inside each MPI process, heavy loops are parallelised with OpenMP.
- The number of threads is set via `OMP_NUM_THREADS` or the LSF affinity
  options.

### MPI + CUDA (CPU–GPU hybrid)

- Same 2D domain decomposition as in the MPI code.
- Coefficients \(a_{ij}, b_{ij}\) and RHS are assembled on CPU and copied
  once to GPU (`cudaMemcpy`).
- On each PCG iteration:
  - **GPU kernels**:
    - `apply_A_kernel` – computes \(Ap\) on the local block;
    - `apply_Dinv_kernel` – applies diagonal preconditioner \(D^{-1}\);
    - `dot_kernel` – partial sums for scalar products.
  - **CPU/MPI operations**:
    - halo exchange for the direction vector \(p\);
    - copying vectors between host and device;
    - global reductions with `MPI_Allreduce`;
    - updates of \(w, r, z, p\) and stopping criterion.

Separate timers measure:
- total solver time \(T_{\text{solver}}\);
- GPU kernel time \(T_{\text{GPU}}\);
- host–device copy time \(T_{\text{copy}}\);
- MPI halo exchange time \(T_{\text{MPI}}\);
- preconditioner work on CPU \(T_{\text{prec}}\);
- scalar products and reductions \(T_{\text{dot}}\).

---

## 3. Code structure (short)

The project is written in C++ (C++11/14) with MPI, OpenMP and CUDA.

Typical files (names may differ slightly):

- `poisson_seq.cpp` – sequential version;
- `poisson_omp.cpp` – OpenMP version;
- `poisson_mpi.cpp` – MPI version;
- `poisson_mpi_omp.cpp` – MPI + OpenMP hybrid;
- `poisson_mpi_cuda.cu` / `poisson_mpi_cuda2.cu` – MPI + CUDA versions;
- common utilities:
  - grid and fictitious domain setup;
  - functions `apply_A`, `apply_Dinv`, `gradient_solver_*`,
  - halo exchange (`exchange_halos_2d`), timers, I/O.

See comments in each file for details.

---

## 4. Building

### Prerequisites

- C++ compiler with OpenMP support (e.g. `g++`, `clang++`, `icpc`);
- MPI implementation (OpenMPI, MPICH, Intel MPI, etc.);
- CUDA Toolkit (for GPU version).

### Example build commands

Sequential / OpenMP:

```bash
g++ -O3 -std=c++14 poisson_seq.cpp -o poisson_seq
g++ -O3 -std=c++14 -fopenmp poisson_omp.cpp -o poisson_omp
