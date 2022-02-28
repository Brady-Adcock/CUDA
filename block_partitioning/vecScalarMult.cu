#include <stdio.h>
#include <stdlib.h>
#include "h_vecScalarMult.h"
#include "d_vecScalarMult.h"
#include "wrappers.h"
#include "config.h"     //this file defines the number of threads per blocks and the number of blocks
#define MAXARRAY 1000000000

//prototypes for functions in this file
void initVector(float * array, int length);
int getVectorLength(int argc, char * argv[]);
void compare(float * result1, float * result2, int n, const char * label);


/*
   driver for the vecScalarMult program.  
*/
int main(int argc, char * argv[])
{
    int n = getVectorLength(argc, argv);
    float * h_A = (float *) Malloc(sizeof(float) * n);
    float K;
    float * h_R1 = (float *) Malloc(sizeof(float) * n);
    float * h_R2 = (float *) Malloc(sizeof(float) * n);
    float * h_R3 = (float *) Malloc(sizeof(float) * n);
    float h_time, d_blktime, d_cyctime, speedup;

    printf("Performing vector scalar multiply on GPU using:\n");
    printf("%d blocks and %d threads per block\n", NUMBLOCKS,
           THREADSPERBLOCK);
    //initialize vector and scalar value
    initVector(h_A, n);
    initVector(&K, 1);
    
    //perform the multiply of the vector by a scalar on the CPU
    h_time = h_vecScalarMult(h_A, h_R1, K, n);
    printf("\nTiming\n");
    printf("------\n");
    printf("CPU: \t\t\t\t%f msec\n", h_time);

    //perform the multiply of the vector by a scalar on the GPU
    //use block partitioning
    d_blktime = d_vecScalarMult(h_A, h_R2, K, n, BLOCK);
    //compare GPU and CPU results 
    compare(h_R1, h_R2, n, "blocked");
    printf("GPU (blocked partitioning): \t%f msec\n", d_blktime);
    speedup = h_time/d_blktime;
    printf("Speedup: \t\t\t%f\n", speedup);

    //perform the multiply of the vector by a scalar on the GPU
    //use cyclic partitioning
    d_cyctime = d_vecScalarMult(h_A, h_R3, K, n, CYCLIC);
    //compare GPU and CPU results 
    compare(h_R1, h_R3, n, "cyclic");
    printf("GPU (cyclic partitioning): \t%f msec\n", d_cyctime);
    speedup = h_time/d_cyctime;
    printf("Speedup: \t\t\t%f\n", speedup);

    free(h_A);
    free(h_R1);
    free(h_R2);
    free(h_R3);
}    

/* 
    getVectorLength
    Converts the second command line argument into an
    integer and returns it.  
    If the command line argument is invalid, it prints usage
    information and exits.
*/
int getVectorLength(int argc, char * argv[])
{
    long int length = -1;
    int numThreads = THREADSPERBLOCK * NUMBLOCKS;
    if (argc == 2) length = atol(argv[1]);

    //length must be an integer greater than the total number of threads
    //length can't be larger than MAXARRAY or program runs out of memory
    if ((length == -1) || (length > MAXARRAY) || (length <= numThreads))
    {
        printf("\nThis program randomly generates a vector of floats\n");
        printf("and a constant value, and multiplies each element of the\n");
        printf("vector by the constant. It performs the multiplication on\n");
        printf("the CPU and the GPU. The product on the GPU is performed\n");
        printf("twice, once using block partitioning and once using cyclic\n");
        printf("partitioning. The program verifies the GPU results by\n");
        printf("comparing them to the CPU results and outputs the times\n");
        printf("it takes to perform each multiplication.\n");
        printf("usage: vecScalarMult <n>\n");
        printf("       <n> size of the vector\n");
        printf("       <n> must be greater than %d\n", numThreads);
        printf("       <n> must be not be greater than %d\n\n", MAXARRAY);
        exit(EXIT_FAILURE);
    }
    return length;
}

/* 
    initVector
    Initializes an array of floats of size
    length to random values between 0 and 99,
    inclusive.
*/
void initVector(float * array, int length)
{
    int i;
    for (i = 0; i < length; i++)
    {
        array[i] = (float)(rand() % 100);
    }
}

/*
    compare
    Compares the values in two vectors and outputs an
    error message and exits if the values do not match.
    result1, result2 - float vectors
    n - length of each vector
    label - string to use in the output message if an error occurs
*/
void compare(float * result1, float * result2, int n, const char * label)
{
    int i;
    for (i = 0; i < n; i++)
    {
        if (result1[i] != result2[i])
        {
            printf("%s partitioning results do not match CPU results.\n", label);
            printf("cpu result[%d]: %f, gpu: result[%d]: %f\n", 
                   i, result1[i], i, result2[i]);
            exit(EXIT_FAILURE);
        }
    }
}

