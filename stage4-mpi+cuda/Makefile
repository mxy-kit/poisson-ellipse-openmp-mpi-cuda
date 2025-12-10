# ==============================
# Makefile for MPI + CUDA (SpectrumMPI)
# ==============================


ARCH      ?= sm_60        
HOST_COMP ?= mpicxx     
NVCC      = nvcc
LINKER    = mpicxx      


NVCCFLAGS = -O3 -std=c++11 -arch=$(ARCH) -ccbin $(HOST_COMP)
LDFLAGS  = -O3 -std=c++11 -L/usr/local/cuda-10.2/lib64 -lcudart -lm

TARGET = poisson_mpi_cuda
OBJS   = poisson_mpi_cuda.o

all: $(TARGET)


%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@


$(TARGET): $(OBJS)
	$(LINKER) $(LDFLAGS) $(OBJS) -o $(TARGET)

clean:
	rm -f $(TARGET) $(OBJS)

