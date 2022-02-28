#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "d_vecAdd.h"

//use this as the size of your blocks (number of threads per block)
#define BLOCKDIM 512 

__global__ void d_vecAddKernel(float * d_A, float * d_B, float * d_C, int n);

/*  d_vecAdd
    Performs the vector add on the GPU (the device).
    A and B are pointers to two vectors to add together.
    The result is stored in the vector pointed to by C.
    n is the length of the vectors.

    returns the amount of time it takes to perform the
    vector add 
*/
float d_vecAdd(float* A, float* B, float* C, int n)
{
    float gpuMsecTime;
    cudaEvent_t start_gpu, stop_gpu;

    //time the sum of the two vectors
    CHECK(cudaEventCreate(&start_gpu));
    CHECK(cudaEventCreate(&stop_gpu));
    CHECK(cudaEventRecord(start_gpu));

    //missing code goes here
    //1) create vectors on the device
    	

	long long size = n * sizeof(float);
	float *d_A, *d_B, *d_C;

	CHECK(cudaMalloc((void **) &d_A, size));
	CHECK(cudaMalloc((void **) &d_B, size));
	CHECK(cudaMalloc((void **) &d_C, size));

    //2) copy A and B vectors into device vectors
    

	CHECK(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

    //3) launch the kernel
    
	dim3 DimGrid(ceil(n/BLOCKDIM), 1,1);
	if(n % BLOCKDIM) DimGrid.x++;	
	dim3 DimBlock(BLOCKDIM, 1, 1);
	d_vecAddKernel<<<DimGrid, DimBlock>>>(d_A, d_B, d_C, n);
    
    //4) copy the result vector into the C vector
   	
	CHECK(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));	

    //5) free space allocated for vectors on the device
    
	CHECK(cudaFree(d_A));
	CHECK(cudaFree(d_B));
	CHECK(cudaFree(d_C));

    //Don't forget to use the CHECK macro on your cuda calls
    CHECK(cudaEventRecord(stop_gpu));
    CHECK(cudaEventSynchronize(stop_gpu));
    CHECK(cudaEventElapsedTime(&gpuMsecTime, start_gpu, stop_gpu));
    return gpuMsecTime;
}

/*  
    d_vecAddKernel
    This function contains the kernel code. This code will be
    executed by every thread created by the kernel launch.
    d_A and d_B are pointers to two vectors on the device to add together.
    The result is stored in the vector pointed to by d_C.
    n is the length of the vectors.
*/
__global__ void d_vecAddKernel(float * d_A, float * d_B, float * d_C, int n)
{
    //add the missing body
	int i = blockIdx.x * blockDim.x  + threadIdx.x;
	if(i < n){
		d_C[i] = d_A[i] + d_B[i];
	}
}      

