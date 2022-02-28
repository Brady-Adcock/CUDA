#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "h_vecAdd.h"

/* h_vecAdd
   Performs the vector add on the CPU (the host).
   A and B are pointers to two vectors to add together.
   The result is stored in the vector pointed to by C.
   n is the length of the vectors.

   returns the amount of time it takes to perform the
   vector add
*/
float h_vecAdd(float* A, float* B, float* C, int n)
{
    cudaEvent_t start_cpu, stop_cpu;
    float cpuMsecTime = -1;

    //Use cuda functions to do the timing 
    //create event objects
    CHECK(cudaEventCreate(&start_cpu));  
    CHECK(cudaEventCreate(&stop_cpu));
    //record the starting time
    CHECK(cudaEventRecord(start_cpu));   

    int i;
    for (i = 0; i < n; i++)
    {
        C[i] = A[i] + B[i];
    }
   
    //record the ending time and wait for event to complete
    CHECK(cudaEventRecord(stop_cpu));
    CHECK(cudaEventSynchronize(stop_cpu)); 
    //calculate the elapsed time between the two events 
    CHECK(cudaEventElapsedTime(&cpuMsecTime, start_cpu, stop_cpu));
    return cpuMsecTime;
}

