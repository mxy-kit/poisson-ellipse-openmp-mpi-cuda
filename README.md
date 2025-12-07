# 2D Poisson Equation on an Elliptic Domain (MPI + OpenMP + CUDA)

This project implements and studies parallel algorithms for solving the
2D Poisson equation on an elliptic domain using the fictitious domain
method and a preconditioned Conjugate Gradient (PCG) solver.

The following implementations are compared:

- Sequential CPU version
- OpenMP (shared–memory)
- MPI (distributed–memory, 2D domain decomposition)
- MPI + OpenMP (hybrid CPU version)
- MPI + CUDA (hybrid CPU–GPU version, one GPU per MPI rank)

The code was developed and tested on the **Polus** cluster (MSU), but can be
adapted to any Linux cluster with MPI and CUDA.

---

## 1. Problem statement

We solve the Poisson equation

$-\Delta u(x,y) = f(x,y)$

in the domain

$D = \{(x,y): x^2 + 4y^2 < 1\}$,

embedded into the rectangular box

$\Omega = [A_1,B_1]\times[A_2,B_2] = [-1,1]\times[-0.6,0.6]$,

with homogeneous Dirichlet boundary conditions

$u(x,y) = 0$ on $\partial D$.

An analytical solution

$u(x,y) = \frac{1}{10}\bigl(1 - x^2 - 4y^2\bigr)$

is used to control the accuracy of the numerical solution.

### Fictitious domain method

The PDE is extended to the whole rectangle $\Omega$ and a piecewise
constant coefficient $k(x,y)$ is used in

$-\nabla\cdot\bigl(k(x,y)\nabla u(x,y)\bigr) = f(x,y)$:

- inside the ellipse $D$: $k(x,y)=1$;
- outside $D$: $k(x,y)=1/\varepsilon$, where  
  $\varepsilon = \max(h_x,h_y)^2$ and $h_x,h_y$ are grid steps.

The rectangle is discretised by a uniform grid of size $M\times N$.
A finite–difference scheme leads to a large sparse linear system which is
solved by PCG with a diagonal preconditioner.

---

## 2. Implemented methods

All versions use the same numerical scheme; they differ only in how
parallelism is organised.

### Sequential CPU

- Builds the full grid on $\Omega$.
- Computes fictitious–domain coefficients $a_{ij}, b_{ij}$ and the right-hand side.
- Runs PCG with a diagonal preconditioner until  
  $\lVert w^{k+1}-w^k\rVert_2 < 10^{-6}$.

The sequential runtime $T_{\text{seq}}$ is used as a reference to compute
speedup $S = T_{\text{seq}}/T$.

### OpenMP

- Shared–memory parallelisation using `#pragma omp parallel for`
  (and `reduction(+:...)` where needed).
- Parallelised parts:
  - computation of $a_{ij}, b_{ij}, f_{ij}$;
  - application of the operator $A$;
  - application of $D^{-1}$;
  - updates of vectors $w,r,z,p$ and norms.

### MPI (2D domain decomposition)

- Internal nodes $(1..M-1)\times(1..N-1)$ are split into a 2D grid of
  subdomains $P_x\times P_y$ (`choose_process_grid`).
- Each MPI process owns a rectangular block plus halo layers.
- Halo exchange uses `MPI_Sendrecv` before applying $A$.
- Global inner products and norms are computed via `MPI_Allreduce`.
- The PCG algorithm is implemented in `gradient_solver_mpi`.

### MPI + OpenMP (CPU hybrid)

- Same 2D MPI decomposition and communication as in the pure MPI code.
- Inside each MPI process, heavy loops are parallelised with OpenMP:
  - local setup of $a_{ij}, b_{ij}, f_{ij}$,
  - application of $A$ and $D^{-1}$,
  - vector updates and local contributions to norms.
- The number of threads is set via `OMP_NUM_THREADS` or LSF affinity options.

### MPI + CUDA (CPU–GPU hybrid)

- Same 2D MPI decomposition as above.
- Coefficients $a_{ij}, b_{ij}$ and RHS are assembled on CPU and copied once
  to GPU arrays.
- On each PCG iteration:
  - **GPU kernels**:
    - `apply_A_kernel` computes $Ap$;
    - `apply_Dinv_kernel` applies the diagonal preconditioner $D^{-1}$;
    - `dot_kernel` computes partial sums for scalar products.
  - **CPU/MPI operations**:
    - halo exchange for the direction vector $p$;
    - host–device copies of $p$, $Ap$, $r$, $z$ where needed;
    - global reductions with `MPI_Allreduce`;
    - updates of $w, r, z, p$ and stopping criterion.

Separate timers measure

- total solver time $T_{\text{solver}}$,
- GPU kernel time $T_{\text{GPU}}$,
- host–device copy time $T_{\text{copy}}$,
- MPI halo exchange time $T_{\text{MPI}}$,
- preconditioner work on CPU $T_{\text{prec}}$,
- scalar products and reductions $T_{\text{dot}}$.

---

## 3. Code structure (short)

The project is written in C++ (C++11/14) with MPI, OpenMP and CUDA.

Typical files:

- `poisson_seq.cpp` – sequential version  
- `poisson_omp.cpp` – OpenMP version  
- `poisson_mpi.cpp` – MPI version  
- `poisson_mpi_omp.cpp` – MPI + OpenMP hybrid  
- `poisson_mpi_cuda.cu`, `poisson_mpi_cuda2.cu` – MPI + CUDA versions  

Utility functions implement grid setup, fictitious domain coefficients,
halo exchange, PCG iterations and timing.

---

## 4. Building

**Prerequisites**

- C++ compiler with OpenMP support (e.g. `g++`, `clang++`, `icpc`)
- MPI implementation (OpenMPI, MPICH, Intel MPI, etc.)
- CUDA Toolkit (for GPU version)

**Examples**

```bash
# sequential
g++ -O3 -std=c++14 poisson_seq.cpp -o poisson_seq

# OpenMP
g++ -O3 -std=c++14 -fopenmp poisson_omp.cpp -o poisson_omp

# MPI
mpicxx -O3 -std=c++14 -fopenmp poisson_mpi.cpp -o poisson_mpi

# MPI + OpenMP
mpicxx -O3 -std=c++14 -fopenmp poisson_mpi_omp.cpp -o poisson_mpi_omp

# MPI + CUDA (sketch)
mpicxx -O3 -std=c++14 -c mpi_part.cpp
nvcc   -O3 -std=c++14 poisson_mpi_cuda.cu mpi_part.o -o poisson_mpi_cuda
