#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
//config.h defines the TILE_WIDTH 
//and the constants: SIMPLE, TILED, TILED2
//that indicate which kernel to launch
#include "config.h" 
#include "d_matMultiply.h"

//prototypes for kernels in this file
__global__ 
void d_matMultiplySimpleKernel(float * d_matrixM, float * d_matrixN, 
                               float * d_result, int width);

__global__ 
void d_matMultiplyTiledKernel(float * d_matrixM, float * d_matrixN, 
                              float * d_result, int width);

__global__ 
void d_matMultiplyTiled2Kernel(float * d_matrixM, float * d_matrixN, 
                               float * d_result, int width);

/*  d_matMultiply
    This function prepares and invokes a kernel to perform
    matrix multiplication (matrixM X matrixN) on the GPU.   
    The matrices have been linearized so each array is
    1D and contains width * width elements.
    Inputs:
    matrixM - points to matrixM data
    matrixN - points to matrixN data
    result - points to the matrix to hold the result
    width - width and height of the input and result matrices
    which - indicates which kernel to use (SIMPLE, TILED, TILED2)
*/
float d_matMultiply(float * matrixM, float * matrixN, float * result, 
                    int width, int which)
{
    cudaEvent_t start_gpu, stop_gpu;
    float gpuMsecTime = -1;

    //time the sum
    CHECK(cudaEventCreate(&start_gpu));
    CHECK(cudaEventCreate(&stop_gpu));
    CHECK(cudaEventRecord(start_gpu));

    //Your work goes here
    //kernel calls provided but you need to write the code for the
    //memory allocations, etc. and define the grid and the block
    //Use TILE_SIZE (defined in config.h)


    float * Mcuda;
    float * Ncuda; 
    float * Rcuda;
    int size = width*width*sizeof(float);
    CHECK(cudaMalloc((void **) &Mcuda, size));
    CHECK(cudaMalloc((void **) &Ncuda, size));
    CHECK(cudaMalloc((void **) &Rcuda, size));
     
    CHECK(cudaMemcpy(Mcuda, matrixM, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Ncuda, matrixN, size, cudaMemcpyHostToDevice));

    if (which == SIMPLE)
    {
	    dim3 grid(ceil(1.0 * width/TILE_WIDTH), ceil(1.0 * width/TILE_WIDTH),1);
    dim3 block(TILE_WIDTH, TILE_WIDTH);

        d_matMultiplySimpleKernel<<<grid, block>>>(Mcuda, Ncuda, Rcuda, width);
    }
    else if (which == TILED)
    {
dim3 grid(ceil(1.0 * width/TILE_WIDTH), ceil(1.0 * width/TILE_WIDTH),1);
    dim3 block(TILE_WIDTH, TILE_WIDTH);

        d_matMultiplyTiledKernel<<<grid, block>>>(Mcuda, Ncuda, Rcuda, width);
    }                                             
    else if (which == TILED2)
    {

dim3 grid(ceil(1.0 * width/TILE_WIDTH), ceil(1.0 * width/TILE_WIDTH),1);
    dim3 block(ceil(1.0 * TILE_WIDTH/2), TILE_WIDTH);
        d_matMultiplyTiled2Kernel<<<grid, block>>>(Mcuda, Ncuda, Rcuda, width);
    }

    CHECK(cudaMemcpy(result, Rcuda, size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(Mcuda));
    CHECK(cudaFree(Ncuda));
    CHECK(cudaFree(Rcuda));

    CHECK(cudaEventRecord(stop_gpu));
    CHECK(cudaEventSynchronize(stop_gpu));
    CHECK(cudaEventElapsedTime(&gpuMsecTime, start_gpu, stop_gpu));
    return gpuMsecTime;
}

/*  
    d_matMultiplySimpleKernel
    This kernel performs matrix multiplication of matrixM and matrixN
    (d_matrixM X d_matrixN) and stores the result in d_result.
    All three matrices are of size width by width and have been linearized.
    Each thread calculates one output element.  All of the elements
    needed for the dot-product calculation are accessed from global
    memory.
    Inputs:
    d_matrixM - pointer to the array containing matrixM
    d_matrixN - pointer to the array containing matrixN
    d_result - pointer to the array in the global memory to hold the result
               of the matrix multiply
    width - width and height of the matrices
*/
__global__ void d_matMultiplySimpleKernel(float * d_matrixM, float * d_matrixN,
                                          float * d_result, int width) 
{

	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	float result = 0; // dont cuck ourselfs with global memeory
	
	if(row >= width || col >= width){
		return;
	}
	
	for(int i = 0; i < width; i++){
		result +=  d_matrixM[row * width + i] * d_matrixN[width * i + col];
	}
	d_result[row * width + col] = result;
}      

/*  
    d_matMultiplyTiledKernel
    This kernel performs matrix multiplication of matrixM and matrixN
    (d_matrixM X d_matrixN) and stores the result in d_result.
    All three matrices are of size width by width and have been linearized.
    Each thread calculates one output element.  Each thread in
    a block cooperates in loading a tile of matrixN and matrixM elements into 
    shared memory and then performs the dot-product calculation using
    the values in the shared memory.  When the threads are finished
    with the current tile, they then load the next tile that is needed.
    At the end, all threads in a block have calculated the results
    of TILE_SIZE by TILE_SIZE output elements.
    Inputs:
    d_matrixM - pointer to the array containing matrixM
    d_matrixN - pointer to the array containing matrixN
    d_result - pointer to the array in the global memory to hold the result
               of the matrix multiply
    width - width and height of the matrices
*/
__global__ 
void d_matMultiplyTiledKernel(float * d_matrixM, float * d_matrixN,
                              float * d_result, int width) 
{
	__shared__ float Ms[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Ns[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * TILE_WIDTH + ty;
	int col = bx * TILE_WIDTH + tx;
	float result = 0;

	for(int i = 0; i < ceil(1.0 * width/TILE_WIDTH); i++){
		if(row < width && i*TILE_WIDTH + tx < width){
			Ms[ty][tx] = d_matrixM[row * width + i * TILE_WIDTH + tx];
		}else{
			Ms[ty][tx] = 0;
		}

		if(col < width && i*TILE_WIDTH + ty < width){
			Ns[ty][tx] = d_matrixN[(i * TILE_WIDTH + ty) * width + col];
		}else{
			Ns[ty][tx] = 0;
		}

		__syncthreads();

		for(int j = 0; j < TILE_WIDTH; j++){
			result += Ms[ty][j] * Ns[j][tx];
		}
		
		__syncthreads();
	}
	if(row < width && col < width){
		d_result[row * width + col] = result;
	}


}      

/*  
    d_matMultiplyTiled2Kernel
    This kernel performs matrix multiplication of matrixM and matrixN
    (d_matrixM X d_matrixN) and stores the result in d_result.
    All three matrices are of size width by width and have been linearized.
    Each thread in a block cooperates in loading a tile of matrixN and 
    matrixM elements into shared memory and then performs the dot-product 
    calculation using the values in the shared memory.  Every thread in 
    the thread block computes 2 results using the values in the shared
    memory.  At the end, all threads in a block have calculated the results
    for TILE_SIZE by TILE_SIZE output elements.
    This implementation is described on page 128 in the textbook.
    Inputs:
    d_matrixM - pointer to the array containing matrixM
    d_matrixN - pointer to the array containing matrixN
    d_result - pointer to the array in the global memory to hold the result
               of the matrix multiply
    width - width and height of the matrices
*/
__global__ 
void d_matMultiplyTiled2Kernel(float * d_matrixM, float * d_matrixN,
                               float * d_result, int width) 
{
	__shared__ float Ms[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Ns[TILE_WIDTH][TILE_WIDTH];

        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        int row = by * TILE_WIDTH + ty;
        int col = (bx * blockDim.x + tx)*2;
	float result = 0;
	float r1 = 0;
	

	for(int i = 0; i < ceil(1.0 * width/TILE_WIDTH); i++){
                
                if(2*tx < TILE_WIDTH){
                        if(col < width && i*TILE_WIDTH + ty < width){
                                Ns[ty][2*tx] = d_matrixN[ (i * TILE_WIDTH + ty) * width + col ];
                        }else{
                                Ns[ty][2*tx] = 0;
                        }
                }
                if(2*tx + 1 < TILE_WIDTH){
                        if(col + 1 < width && i*TILE_WIDTH + ty < width){
                                Ns[ty][2*tx+1] = d_matrixN[ (i * TILE_WIDTH + ty) * width + col + 1];
                        }else{
                                Ns[ty][2*tx+1] = 0;
                        }
                }
		
                if(2*tx < TILE_WIDTH){
                        if(row < width && i*TILE_WIDTH + 2*tx < width){
                                Ms[ty][2*tx] = d_matrixM[row * width + i * TILE_WIDTH + 2*tx ];
                        }else{
                                Ms[ty][2*tx] = 0;
                        }
                }
                if(2*tx + 1 < TILE_WIDTH){
                        if(row < width && i*TILE_WIDTH + 2*tx + 1 < width){
                                Ms[ty][2*tx+1] = d_matrixM[row * width + i * TILE_WIDTH + 2*tx + 1 ];
                        }else{
                                Ms[ty][2*tx+1] = 0;
                        }
                }

		__syncthreads();



                for(int j = 0; j < TILE_WIDTH; j++){
                        
			result += Ms[ty][j] * Ns[j][2*tx];
			r1 += Ms[ty][j] * Ns[j][2*tx+1];
			//resultP += Ns[ty][j] * MsP[j][tx];
			//r2 += Ns[ty][j] * MsP[j][tx+1];
                }
		__syncthreads();

        }
       if(row < width && col < width){
                d_result[row * width + col] = result;
       }
       if(row < width && col+1 < width){
		d_result[row * width + col + 1] = r1;
       }
}      

