#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "h_matMultiply.h"

//prototype for function local to this file
void matMultiplyOnCPU(float * h_matrixM, float * h_matrixN, 
                      float * h_result, int width);

/*  h_matMultiply
    This function returns the amount of time it takes to perform
    a matrix multiply on the CPU.
    Inputs:
    h_matrixM - points to the one matrix to use in the multiply
    h_matrixN - points to the the other matrix to use in the multiply
    h_result - points to the matrix to hold the result
    width - x and y dimension of the matrices (width by width)

    returns the amount of time it takes to perform the
    matrix multiply
*/
float h_matMultiply(float * h_matrixM, float * h_matrixN, 
                    float * h_result, int width) 
{
    cudaEvent_t start_cpu, stop_cpu;
    float cpuMsecTime = -1;

    //Use CUDA functions to do the timing 
    //create event objects
    CHECK(cudaEventCreate(&start_cpu));  
    CHECK(cudaEventCreate(&stop_cpu));
    //record the starting time
    CHECK(cudaEventRecord(start_cpu));   
    
    //call function that does the actual work
    matMultiplyOnCPU(h_matrixM, h_matrixN, h_result, width);
   
    //record the ending time and wait for event to complete
    CHECK(cudaEventRecord(stop_cpu));
    CHECK(cudaEventSynchronize(stop_cpu)); 

    //calculate the elapsed time between the two events 
    CHECK(cudaEventElapsedTime(&cpuMsecTime, start_cpu, stop_cpu));
    return cpuMsecTime;
}

/*  matMultiplyOnCPU
    This function performs the matrix multiply on the CPU.  
    Inputs:
    h_matrixM - points to the one of the matrices to use in the multiply
    h_matrixN - points to the one of the matrices to use in the multiply
    h_result - points to the matrix to hold the result
    width - both the x and y dimension of the matrices (width by width)

    modifies the h_result matrix
*/
void matMultiplyOnCPU(float * h_matrixM, float * h_matrixN,
                      float * h_result, int width)
{
    int i, j, k; 
    float cvalue;
    //go through all rows of M
    for (i = 0; i < width; i++)
    {
        //go through all columns of N
        for (j = 0; j < width; j++)
        {
            cvalue = 0;
            //compute dot product
            for (k = 0; k < width; k++)
            {
                //matrices have been linearized so use 2D indices
                //to calculate 1D index
                cvalue += h_matrixM[i * width + k] * h_matrixN[k * width + j];
            }
            h_result[i * width + j] = cvalue;
        }
    }
}
