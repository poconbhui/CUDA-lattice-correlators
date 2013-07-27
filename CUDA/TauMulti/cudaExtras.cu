#ifndef CUDA_EXTRAS_CU
#define CUDA_EXTRAS_CU
//////////////////////////////////////////////////////////////
//
//	Extra little bits and scraps to make life easier
//
//////////////////////////////////////////////////////////////

#include "cudaExtras.h"

std::string cuComplexftoString(cuComplex vect)
	{
	char buffer[20];
	
	sprintf(buffer, "(%.3f,%.3f)",cuCrealf(vect),cuCimagf(vect));
	
	return std::string(buffer);
	}

std::string printSize(int MemTotal)
{
char buffer[20];

if(MemTotal > 1024*1024*1024)
	{
	sprintf(buffer, "%.2f GB",MemTotal/(1024.0*1024.0*1024.0));
	}
else if(MemTotal > 1024*1024)
	{
	sprintf(buffer, "%.2f MB",MemTotal/(1024.0*1024.0));
	}
else
	{
	sprintf(buffer, "%.2f kB",MemTotal/(1024.0));
	}
	
	return std::string(buffer);
}
#endif
