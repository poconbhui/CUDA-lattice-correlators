#ifndef TAU_KERNELS_H
#define TAU_KERNELS_H

///////////////////////////////////////////////////////
//
//	This is for defining CUDA functions and
//	C wrappers for CUDA calls
//
///////////////////////////////////////////////////////


#include <cuda.h>
#include <cuComplex.h>
#include "defines.h"

void Multiply(dim3 dimGrid, dim3 dimBlock, cuFloatComplex *I, cuFloatComplex *T, cuFloatComplex *X);
__global__ void MultiplyKernel(cuFloatComplex *I, cuFloatComplex *T, cuFloatComplex *X);

void Trace(dim3 dimGrid, dim3 dimBlock, cuFloatComplex *C, cuFloatComplex *X, int d);
__global__ void TraceKernel(cuFloatComplex *C, cuFloatComplex *X, int d);

#endif
