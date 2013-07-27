#ifndef CUDA_EXTRAS_H
#define CUDA_EXTRAS_H

//////////////////////////////////////////////////////////////
//
//	Extra little bits and scraps to make life easier
//
//////////////////////////////////////////////////////////////


#include <cuda.h>
#include <cuComplex.h>

#include <stdio.h>
#include <string>

#include "defines.h"

std::string cuComplexftoString(cuComplex vect);

std::string printSize(int MemTotal);


#endif
