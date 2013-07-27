//Display some infor about the card and machine


//#include <cuda.h>

#include <iostream>
#include <iomanip>

//http://stackoverflow.com/questions/5689028/how-to-get-card-specs-programatically-in-cuda/5689133#5689133
void CUDA_Info()
{
using namespace std;
    const int kb = 1024;
    const int mb = kb * kb;
    wcout << endl << "NBody.GPU" << endl << "=========" << endl << endl;

#ifdef CUDART_VERSION
    wcout << "CUDA version:   v" << CUDART_VERSION << endl;  
#endif  

#ifdef THRUST_MAJOR_VERSION
    wcout << "Thrust version: v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << endl << endl; 
#endif

    int devCount;
    cudaGetDeviceCount(&devCount);
    wcout << "CUDA Devices: " << endl << endl;

    for(int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        wcout << i << ": " << props.name << ": " << props.major << "." << props.minor << endl;
        wcout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << endl;
        wcout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << endl;
        wcout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
        wcout << "  Block registers: " << props.regsPerBlock << endl << endl;

        wcout << "  Warp size:         " << props.warpSize << endl;
        wcout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
        wcout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << endl;
        wcout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << endl;
        wcout << endl;
    }
}
