Computing Two Point Correlators For A Lattice QCD Theory On Graphics Processor Units
====================================================================================

My final year project for Theoretical Physics.

This project aimed to write CUDA kernels for computing two point correlator
functions. For this, I explored implementing matrix multipliation kernels.
I managed to beat the CUBLAS implementation on the computer I was working on
by 35% for batch multiplication jobs.

For the particular problem I was working on, I could also fold some sparse
matrix multiplications into my kernels, further improving overall performance.

Finally, by casting the final part of the problem from many matrix trace
operations to several matrix multiplication problems, I improved performance
in this task by two orders of magnitude.

The code is available in `CUDA/TauMulti`. The writeup is available in `report`
