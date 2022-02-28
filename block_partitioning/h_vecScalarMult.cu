#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "h_vecScalarMult.h"

/*  h_vecScalarMult
    Uses the CPU (the host) to multiply a vector by a scalar.
    A is a pointer to the input vector.
    K is the scalar value to use in the multiply
    The result is stored in the vector pointed to by R.
    n is the length of the vectors.

    returns the amount of time it takes to perform the
    vector scalar multiply
*/
float h_vecScalarMult(float* A, float * R, float K, int n)
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
        R[i] = A[i] * K;
    }
   
    //record the ending time and wait for event to complete
    CHECK(cudaEventRecord(stop_cpu));
    CHECK(cudaEventSynchronize(stop_cpu)); 
    //calculate the elapsed time between the two events 
    CHECK(cudaEventElapsedTime(&cpuMsecTime, start_cpu, stop_cpu));
    return cpuMsecTime;
}

