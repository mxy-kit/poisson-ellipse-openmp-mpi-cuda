# ==============================
# Makefile for MPI + CUDA (SpectrumMPI)
# ==============================


ARCH      ?= sm_60      
HOST_COMP ?= mpicc       

NVCC   = nvcc
LINKER = mpicxx         


NVCCFLAGS = -O3 -std=c++11 -arch=$(ARCH) -ccbin $(HOST_COMP)
LDFLAGS  = -O3 -std=c++11 -L/usr/local/cuda/lib64 -lcudart -lm

TARGET = poisson_mpi_cuda2
OBJS   = poisson_mpi_cuda2.o

all: $(TARGET)


%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@


$(TARGET): $(OBJS)
	$(LINKER) $(LDFLAGS) $(OBJS) -o $(TARGET)

clean:
	rm -f $(TARGET) $(OBJS)

