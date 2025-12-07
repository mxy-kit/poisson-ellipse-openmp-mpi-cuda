# ==============================
# Makefile for MPI + CUDA (SpectrumMPI)
# ==============================

# 题目要求的两个变量 —— 一定要有
ARCH      ?= sm_60       # 或 sm_35，看老师要求
HOST_COMP ?= mpicc       # host 端编译器，传给 nvcc

NVCC   = nvcc
LINKER = mpicxx          # 只在链接阶段用 mpicxx，不用 CXX 这个名字，避免环境变量干扰

# nvcc 编译 .cu -> .o 时用 HOST_COMP 作为 host compiler
NVCCFLAGS = -O3 -std=c++11 -arch=$(ARCH) -ccbin $(HOST_COMP)
LDFLAGS  = -O3 -std=c++11 -L/usr/local/cuda/lib64 -lcudart -lm

TARGET = poisson_mpi_cuda2
OBJS   = poisson_mpi_cuda2.o

all: $(TARGET)

# 1) .cu -> .o 由 nvcc 完成
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# 2) 最终链接 —— 一定要用 mpicxx（LINKER），这样 MPI_* 和 ompi_* 都有定义
$(TARGET): $(OBJS)
	$(LINKER) $(LDFLAGS) $(OBJS) -o $(TARGET)

clean:
	rm -f $(TARGET) $(OBJS)

