#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "d_colorToGreyscale.h"
#include "CHECK.h"

#define CHANNELS 3
#define BLOCKSIZE 32
__global__ void d_colorToGreyscaleKernel(unsigned char *, unsigned char *,
                                         int, int);
/*
   d_colorToGreyscale
   Performs the greyscale of an image on the GPU.
   Pout array is filled with the greyscale of each pixel.
   Pin array contains the color pixels.
   width and height are the dimensions of the image.
*/
float d_colorToGreyscale(unsigned char * Pout, unsigned char * Pin,
                        int width, int height)
{
    cudaEvent_t start_cpu, stop_cpu;
    float cpuMsecTime = -1;

    //Use cuda functions to do the timing 
    //create event objects
    CHECK(cudaEventCreate(&start_cpu));
    CHECK(cudaEventCreate(&stop_cpu));
    //record the starting time
    CHECK(cudaEventRecord(start_cpu));

    //Your work goes here and in the kernel below
    int size = width * height * sizeof(unsigned char);
    unsigned char * PoutCuda;
    unsigned char * PinCuda;

    CHECK(cudaMalloc((void **) &PoutCuda, size));
    CHECK(cudaMalloc((void **) &PinCuda, size*CHANNELS));

    CHECK(cudaMemcpy(PinCuda, Pin, size*CHANNELS, cudaMemcpyHostToDevice));

    dim3 grid(ceil(width/BLOCKSIZE)+1, ceil(height/BLOCKSIZE)+1, 1);
    dim3 block(BLOCKSIZE, BLOCKSIZE, 1);
    
    d_colorToGreyscaleKernel<<<grid, block>>>(PinCuda, PoutCuda, width, height);

    CHECK(cudaMemcpy(Pout, PoutCuda, size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(PoutCuda)); 
    CHECK(cudaFree(PinCuda));
    //record the ending time and wait for event to complete
    CHECK(cudaEventRecord(stop_cpu));
    CHECK(cudaEventSynchronize(stop_cpu));
    //calculate the elapsed time between the two events 
    CHECK(cudaEventElapsedTime(&cpuMsecTime, start_cpu, stop_cpu));
    return cpuMsecTime;
}

/*
   d_colorToGreyscaleKernel
   Kernel code executed by each thread on its own data when the kernel is
   launched.
   Pout array is filled with the greyscale of each pixel (one element per thread).
   Pin array contains the color pixels.
   width and height are the dimensions of the image.
*/
__global__
void d_colorToGreyscaleKernel(unsigned char * Pin, unsigned char * Pout,
                              int width, int height)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	if( col < width && row < height ){
		int flat = row*width + col;
		int offset = CHANNELS * flat;
		unsigned char r = Pin[offset];
		unsigned char g = Pin[offset+1];
		unsigned char b = Pin[offset+2];
		Pout[flat] = 0.21f*r + 0.71f*g + 0.07f*b;
	}
}
