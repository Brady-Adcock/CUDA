#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "config.h" //defines the number of threads per block and the number of blocks
#include "d_vecScalarMult.h"

//put the prototypes for the two kernels here
__global__ void d_CyclicPartition(float* in, float* out, float K, int size);
__global__ void d_BlockPartition(float* in, float* out, float K, int size);

/*  d_vecScalarMult
    Multiples a vector by a scalar using the GPU (the device).
    A is a pointer to the input vector.
    K is the scalar to use in the multiply.
    The result is stored in the vector pointed to by R.
    n is the length of the vectors.
    which indicates whether the work should be distributed among the threads
    using BLOCK partitioning or CYCLIC partitioning.

    returns the amount of time it takes to perform the
    vector scalar multiply
*/
float d_vecScalarMult(float* A, float * R, float K, int n, int which)
{
    cudaEvent_t start_gpu, stop_gpu;
    float gpuMsecTime = -1;

    //time the sum
    CHECK(cudaEventCreate(&start_gpu));
    CHECK(cudaEventCreate(&stop_gpu));
    CHECK(cudaEventRecord(start_gpu));

    //your code to prepare for and invoke the kernel goes here
    int size = n*sizeof(float);
    float* result;
    float* input;
    CHECK(cudaMalloc((void **) &result, size ));
    CHECK(cudaMalloc((void **) &input, size ));

    CHECK(cudaMemcpy(input, A, size, cudaMemcpyHostToDevice));
    
    dim3 grid(NUMBLOCKS, 1, 1);
    dim3 blocks(THREADSPERBLOCK, 1, 1);

    
    
    if (which == BLOCK)
       //call kernel that uses block partitioning  
	    d_BlockPartition<<<grid, blocks>>>(input, result, K, n);
    else
       //call kernel that uses cyclic partitioning  
	    d_CyclicPartition<<<grid, blocks>>>(input, result, K, n);

    CHECK(cudaMemcpy(R, result, size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(input));
    CHECK(cudaFree(result));

    CHECK(cudaEventRecord(stop_gpu));
    CHECK(cudaEventSynchronize(stop_gpu));
    CHECK(cudaEventElapsedTime(&gpuMsecTime, start_gpu, stop_gpu));
    return gpuMsecTime;
}

//put the two kernels here
__global__ void d_BlockPartition(float* in, float* out, float K, int size){
	int threads = NUMBLOCKS*THREADSPERBLOCK;
	int work = ceil(((1.0) * size)/threads);
	int threadpos = blockDim.x  * blockIdx.x + threadIdx.x;
	for(int i = 0; i < work; i++){
		int index = threadpos * work + i;
		if(index >= size){
			break;
		}
		out[index] = in[index]*K;
	}	
}

__global__ void d_CyclicPartition(float* in, float* out, float K, int size){
	int threads = NUMBLOCKS*THREADSPERBLOCK; // total number of threads
	int work = ceil(((1.0) * size)/threads); // amount of work each thread has to do 
	int threadpos = blockDim.x  * blockIdx.x + threadIdx.x; // thread position within gird
	for(int i = 0; i < work; i++){ 
		int index = threadpos + threads * i; // example : t0, when 16 elemnts and kernel launches
		if(index >= size){		  //               4 threads. t0 should work on the
			break;			  //		   following elemnts: 0, 4, 8, 12
		}
		out[index] = in[index]*K;
	}
}


