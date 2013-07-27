#include <iostream>
#include <iomanip>

#include <boost/thread.hpp>

#include "sys/types.h"
#include "sys/sysinfo.h"

struct sysinfo memInfo;

//http://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process/64166#64166
void MachineInfo()
{
using namespace std;
    wcout << endl << "Machine Values" << endl << "=========" << endl << endl;
    sysinfo (&memInfo);
    long long totalVirtualMem = memInfo.totalram;
    //Add other values in next statement to avoid int overflow on right hand side...
    totalVirtualMem += memInfo.totalswap;
    totalVirtualMem *= memInfo.mem_unit;
    totalVirtualMem /= 1024*1024;
    
    wcout << "Total Virtual Memory: " << totalVirtualMem << " mb" << endl;
    
    long long totalPhysMem = memInfo.totalram;
    //Multiply in next statement to avoid int overflow on right hand side...
    totalPhysMem *= memInfo.mem_unit;
    totalPhysMem /= 1024*1024;
    
    wcout << "Total RAM: " << totalPhysMem << " mb" << endl;
    
    int numCPU = sysconf( _SC_NPROCESSORS_ONLN );
    
    wcout << "CPU Cores: " << numCPU << endl;
    
    wcout << "CPU Concurrent Threads " << boost::thread::hardware_concurrency() << endl;
}
