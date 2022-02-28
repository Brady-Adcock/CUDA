#include <stdio.h>
#include <stdlib.h>
#include "h_matMultiply.h"
#include "d_matMultiply.h"
#include "wrappers.h"
//config.h defines the TILE_SIZE
//and the constants: SIMPLE, TILED, TILED2 that 
//indicate which kernel to launch
#include "config.h"     
#define DEBUG 1

//struct to store speedups
typedef struct
{
   char version[10];
   float speedup;
} resultT;

//prototypes for functions in this file
void initMatrix(float * array, int length);
void getDimension(int argc, char * argv[], int *);
void compare(float * result1, float * result2, int width, const char * label);
void printUsage();
//helper functions for debugging
void printResults(float * matrixM, float * matrixN, float * result, int width);
void printMatrix(float * matrix, int width);
void printComparisons(resultT *, int);

/*
   driver for the matMultiply program.  
*/
int main(int argc, char * argv[])
{
    int matrixDim;
    resultT results[4];
    getDimension(argc, argv, &matrixDim);
    float * h_matrixM = (float *) Malloc(sizeof(float) * matrixDim * matrixDim);
    float * h_matrixN = (float *) Malloc(sizeof(float) * matrixDim * matrixDim);
    float * h_result = (float *) Malloc(sizeof(float) * matrixDim * matrixDim);
    float * d_result = (float *) Malloc(sizeof(float) * matrixDim * matrixDim);
    float h_time, d_simpletime, d_tiledtime, d_tiled2time, speedup;

    //initialize matrix and scalar value
    initMatrix(h_matrixM, matrixDim * matrixDim);
    initMatrix(h_matrixN, matrixDim * matrixDim);
   
    //perform the sum of the matrices on the CPU
    h_time = h_matMultiply(h_matrixM, h_matrixN, h_result, matrixDim);
    if (DEBUG)
        printResults(h_matrixM, h_matrixN, h_result, matrixDim);
    printf("\nTiming\n");
    printf("------\n");
    printf("CPU: \t\t\t\t%f msec\n", h_time);
    results[0].speedup = 1.0;
    strcpy(results[0].version, "CPU");

    memset(d_result, 0, matrixDim * matrixDim);
    d_simpletime = d_matMultiply(h_matrixM, h_matrixN, d_result, 
                                 matrixDim, SIMPLE);
    //compare GPU and CPU results 
    compare(h_result, d_result, matrixDim, "simple");
    printf("GPU (simple version): \t\t%f msec\n", d_simpletime);
    speedup = h_time/d_simpletime;
    printf("Speedup: \t\t\t%f\n", speedup);
    results[1].speedup = speedup;
    strcpy(results[1].version, "SIMPLE");
    
    memset(d_result, 0, matrixDim * matrixDim);
    d_tiledtime = d_matMultiply(h_matrixM, h_matrixN, d_result, 
                                 matrixDim, TILED);
    if (DEBUG)
        printResults(h_matrixM, h_matrixN, d_result, matrixDim);
    //compare GPU and CPU results 
    compare(h_result, d_result, matrixDim, "tiled");
    printf("GPU (tiled version): \t\t%f msec\n", d_tiledtime);
    speedup = h_time/d_tiledtime;
    printf("Speedup: \t\t\t%f\n", speedup);
    results[2].speedup = speedup;
    strcpy(results[2].version, "TILED");

    memset(d_result, 0, matrixDim * matrixDim);
    d_tiled2time = d_matMultiply(h_matrixM, h_matrixN, d_result, 
                                 matrixDim, TILED2);
    //compare GPU and CPU results 
    compare(h_result, d_result, matrixDim, "tiled2");
    printf("GPU  (tiled2 version): \t\t%f msec\n", d_tiled2time);
    speedup = h_time/d_tiled2time;
    printf("Speedup: \t\t\t%f\n", speedup);
    results[3].speedup = speedup;
    strcpy(results[3].version, "TILED2");
    printComparisons(results, 4);

    free(h_result);
    free(d_result);
    free(h_matrixM);
    free(h_matrixN);
}    

/*
   printComparions
   This function sorts the speedups and prints the results relative to
   each other.
*/
void printComparisons(resultT * results, int size)
{
   float current, tcurrent;
   char type[10], ttype[10];
   int i, j;

   for (i = 1; i < size; i++)
   {
      current = results[i].speedup;
      strcpy(type, results[i].version); 
      for (j = 0; j < i; j++)
      {
         if (current < results[j].speedup)
         {
  
            tcurrent = results[j].speedup;
            strcpy(ttype, results[j].version);
            results[j].speedup = current;
            strcpy(results[j].version, type);
            current = tcurrent;
            strcpy(type, ttype);
         }
      }
      results[i].speedup = current;
      strcpy(results[i].version, type);
   }
   printf("\nPerformance: ");
   for (i = 0; i < size - 1; i++)
      printf("%s < ", results[i].version);
   printf("%s\n", results[size-1].version);
} 

/* 
    getDimension
    This function parses the command line arguments to get
    the dimension of the matrices. If the command line argument 
    is invalid, it prints usage information and exits.
    Inputs:
    argc - count of the number of command line arguments
    argv - array of command line arguments
    matrixDim - pointer to an int to be set to the matrix dimension
*/
void getDimension(int argc, char * argv[], int * matrixDim)
{
    if (argc != 2) printUsage();
    int dimension = atoi(argv[1]);
    if (dimension < 0) printUsage();
    (*matrixDim) = dimension;
}

/*
    printUsage
    prints usage information and exits
*/
void printUsage()
{
    printf("\nThis program performs the matrix multiply of two arrays\n");
    printf("of size n by n. The value of n is provided as a command line\n");
    printf("argument.\n");
    printf("usage: matMultiply <n>\n");
    printf("       <n> is the height and width of the matrix\n");
    exit(EXIT_FAILURE);
}

/* 
    initMatrix
    Initializes an array of floats of size
    length to random values between 0 and 10.
    Inputs:
    array - pointer to the array to initialize
    length - length of array
*/
void initMatrix(float * array, int length)
{
    int i;
    for (i = 0; i < length; i++)
    {
        array[i] = (float)(rand() % 10);
    }
}

/*
    compare
    Compares the values in two matrices and outputs an
    error message and exits if the values do not match.
    result1, result2 - float matrices
    n - length of each matrix
    label - string to use in the output message if an error occurs
*/
void compare(float * result1, float * result2, int n, const char * label)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        { 
            float diff = abs(result1[i * n + j] - result2[i * n + j]);
            if (diff > 0.01) // 
            {
                printf("%s GPU multiply does not match CPU results.\n", label);
                printf("cpu result[%d, %d]: %f, gpu: result[%d, %d]: %f\n", 
                   i, j, result1[i * n + j], i, j, result2[i * n + j]);
                exit(EXIT_FAILURE);
            }
        }
    }
}

//helper function for debugging
void printResults(float * matrixM, float * matrixN, float * result, int width)
{
    printf("\nmatrixM: \n");
    printMatrix(matrixM, width);
    printf("\nmatrixN: \n");
    printMatrix(matrixN, width);
    printf("\nresult: \n");
    printMatrix(result, width);
}

//helper function for debugging
void printMatrix(float * matrix, int width)
{
    int i, j;
    for (i = 0; i < width; i++)
    {
        for (j = 0; j < width; j++)
        {
            printf("%7.2f ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

