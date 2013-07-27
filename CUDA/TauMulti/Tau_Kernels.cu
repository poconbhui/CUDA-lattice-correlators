#ifndef TAU_KERNELS_CU
#define TAU_KERNELS_CU

////////////////////////////////////////////////////////////////
//
//	This is where the main Cuda kernels are declared
//	along with C++ wrappers, so that C++ files can
//	be compiled which call them. The only difference
//	between calling the Cuda kernel and the C++
//	wrapper is that the grid and block dimensions
//	are defined in the first two arguments of the
//	wrapper. After that, everything is the same.
//
//	The main two are the main multiplication kernel
//	and the trace kernel.
//
////////////////////////////////////////////////////////////////

#include "Tau_Kernels.h"

void Multiply(dim3 dimGrid, dim3 dimBlock, cuFloatComplex *I, cuFloatComplex *T, cuFloatComplex *X)
{
MultiplyKernel<<< dimGrid, dimBlock >>>(I, T, X);
}

__global__ void MultiplyKernel(cuFloatComplex *I, cuFloatComplex *T, cuFloatComplex *X)
{
#ifdef TEST_DEFS
int k;

#ifdef MULTIPLY_DIMS_1
int h = threadIdx.z/C_MATRIX_DIM;
int i = threadIdx.z/2;
int d = threadIdx.y;
int t = threadIdx.x;
int y = gridDim.y;
int x = gridDim.x;

int size = h*C_MATRIX_DIM*DELTA_CHUNKSIZE*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM +
	i*DELTA_CHUNKSIZE*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM +
	d*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM +
	t*PHI_MATRIX_DIM*PHI_MATRIX_DIM +
	y*PHI_MATRIX_DIM +
	x;

#else

int size = 
	blockIdx.y*gridDim.x*blockDim.z*blockDim.y*blockDim.x +
	blockIdx.x*blockDim.z*blockDim.y*blockDim.x +
	threadIdx.z*blockDim.y*blockDim.x + 
	threadIdx.y*blockDim.x + 
	threadIdx.x;

//int size = blockIdx.x*blockDim.x + threadIdx.x;

//I like this sneakey trick.
//There's probably a MUCH better way of doing this though...
int x = size;

int h = static_cast<int>(
			x/(C_MATRIX_DIM*DELTA_CHUNKSIZE*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM)
			);
x -= h*C_MATRIX_DIM*DELTA_CHUNKSIZE*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM;

int i = static_cast<int>(
			x/(DELTA_CHUNKSIZE*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM)
			);
x -= i*DELTA_CHUNKSIZE*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM;
	
int d = static_cast<int>(
			x/(T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM)
			);
x -= d*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM;

int t = static_cast<int>(
			x/(PHI_MATRIX_DIM*PHI_MATRIX_DIM)
			);
x -= t*PHI_MATRIX_DIM*PHI_MATRIX_DIM;

int y = static_cast<int>(
			x/(PHI_MATRIX_DIM)
			);
x -= y*PHI_MATRIX_DIM;

#endif

cuFloatComplex Xtmp = make_cuFloatComplex(0,0);

if(h==0) //T loop
	{
	#pragma unroll
	for(k=0; k<PHI_MATRIX_DIM;k++)
		{
		Xtmp = cuCaddf(Xtmp,
			cuCmulf(
				I[
					i*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM + 
					t*PHI_MATRIX_DIM*PHI_MATRIX_DIM + 
					y*PHI_MATRIX_DIM + 
					k
					],
				T[
					d*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM +
					t*PHI_MATRIX_DIM*PHI_MATRIX_DIM +
					k*PHI_MATRIX_DIM +
					x
					]
				)
			);
		}
	}
else //T^+ loop
	{
	#pragma unroll
	for(k=0; k<PHI_MATRIX_DIM;k++)
		{
		Xtmp = cuCaddf(Xtmp,
			cuCmulf(
				I[
					i*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM + 
					t*PHI_MATRIX_DIM*PHI_MATRIX_DIM + 
					y*PHI_MATRIX_DIM + 
					k
					],
				cuConjf(T[
					d*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM +
					t*PHI_MATRIX_DIM*PHI_MATRIX_DIM +
					x*PHI_MATRIX_DIM +
					k
					]) //complex conjugate transpose
				)
			);
		}
	}

X[size] = Xtmp;
//X[size] = make_cuFloatComplex(x,y);

#endif
}


void Trace(dim3 dimGrid, dim3 dimBlock, cuComplex *C, cuComplex *X, int d)
{
TraceKernel<<< dimGrid, dimBlock >>>(C, X, d);
}

__global__ void TraceKernel(cuComplex *C, cuComplex *X, int d)
{
#ifdef TEST_DEFS
int size = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;

//int size = blockIdx.x*blockDim.x + blockIdx.y*blockDim.y + threadIdx.x;
/*int j = size;
int d_it = (int) j/(C_MATRIX_DIM*C_MATRIX_DIM);
j -= d_it*C_MATRIX_DIM*C_MATRIX_DIM;
int i = (int) j/C_MATRIX_DIM;
j -= i*C_MATRIX_DIM;*/


int d_it = threadIdx.x;
int i = blockIdx.x;
int j = blockIdx.y;


int a, b, t;
int current_pos[2];

cuFloatComplex Ctmp = make_cuFloatComplex(0,0);


#pragma unroll
for(t=0; t<T_CHUNKSIZE; t++) //WORK ON THIS!!
	{
	current_pos[0] = i*DELTA_CHUNKSIZE*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM +
			d_it*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM +
			t*PHI_MATRIX_DIM*PHI_MATRIX_DIM;
	current_pos[1] = j*DELTA_CHUNKSIZE*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM +
			d_it*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM +
			t*PHI_MATRIX_DIM*PHI_MATRIX_DIM;
	
	#pragma unroll
	for(a=0; a<PHI_MATRIX_DIM; a++)
		{
		#pragma unroll
		for(b=0; b<PHI_MATRIX_DIM; b++)
			{
			
			Ctmp = cuCaddf(Ctmp,
				cuCmulf(
					X[
						current_pos[0] +
						b*PHI_MATRIX_DIM +
						a
						],
					X[
						C_MATRIX_DIM*DELTA_CHUNKSIZE*T_CHUNKSIZE*PHI_MATRIX_DIM*PHI_MATRIX_DIM +
						current_pos[1] +
						a*PHI_MATRIX_DIM +
						b
						]
					)
				);
			
			}
		}
	}


C[d*C_MATRIX_DIM*C_MATRIX_DIM + size] = cuCaddf(C[d*C_MATRIX_DIM*C_MATRIX_DIM + size] , Ctmp);
//C[d*C_MATRIX_DIM*C_MATRIX_DIM + size] = make_cuFloatComplex(d,d*C_MATRIX_DIM*C_MATRIX_DIM + size);

#endif
}

#endif
