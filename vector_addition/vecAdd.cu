#include <stdio.h>
#include <stdlib.h>
#include "h_vecAdd.h"
#include "d_vecAdd.h"
#include "wrappers.h"

//prototypes for functions in this file
void initVector(float * array, int length);
int getVectorLength(int argc, char * argv[]);
void compare(float * result1, float * result2, int n);


/*
   driver for the vecAdd program.  
*/
int main(int argc, char * argv[])
{
    int n = getVectorLength(argc, argv);
    float * h_A = (float *) Malloc(sizeof(float) * n);
    float * h_B = (float *) Malloc(sizeof(float) * n);
    float * h_C1 = (float *) Malloc(sizeof(float) * n);
    float * h_C2 = (float *) Malloc(sizeof(float) * n);
    float h_time, d_time, speedup;

    initVector(h_A, n);
    initVector(h_B, n);
    h_time = h_vecAdd(h_A, h_B, h_C1, n);
    d_time = d_vecAdd(h_A, h_B, h_C2, n);
    compare(h_C1, h_C2, n);
    free(h_A);
    free(h_B);
    free(h_C1);
    free(h_C2);

    printf("Time of vector add on CPU: %f msec\n", h_time);
    printf("Time of vector add on GPU: %f msec\n", d_time);
    speedup = h_time/d_time;
    printf("Speedup: %f\n", speedup);
}    


/* 
    getVectorLength
    Converts the second command line argument into an
    integer and returns it.  
*/
int getVectorLength(int argc, char * argv[])
{
    int length;
    if (argc != 2 || (length = atoi(argv[1])) <= 0)
    {
        printf("\nThis program randomly generates two vectors of floats,\n");
        printf("computes the sum of the two vectors on both the CPU and the GPU,\n");
        printf("and outputs the time it takes to perform the sums.\n\n");
        printf("usage: vecAdd <n>\n");
        printf("       <n> size of the vectors\n");
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
*/
void compare(float * result1, float * result2, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        if (result1[i] != result2[i])
        {
	    printf("GPU and CPU results do not match.\n");
            exit(EXIT_FAILURE);
        }
    }
}

