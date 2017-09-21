/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

#ifdef _WIN32
#  define NOMINMAX 
#endif

#define NUM_BANKS 16
#define N_ELEMENTS 16384

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// includes, project
//#include <cutil_inline.h>
#include <helper_cuda.h>

// includes, kernels
__global__ void sum_kernel(float *g_odata, float *g_idata, int n)
{
    int i;

    int tid = threadIdx.x; //Calculate a thread ID based on this thread's position within the block
    //int tid = threadIdx.y * blockDim.x + threadIdx.x; //Another thread ID example for a 2-D thread block
    //int tid = blockIdk.x * blockDim.x + threadIdx.x; //Another thread ID example for assigning unique thread IDs across
    //different blocks

    g_odata[0] = 0;

    //A single thread adds up all 1M array elements serially.
    //This is a poor use of parallel hardware - your job is to increase the number of threads, split up the work, and communicate
    //data between threads as necessary to improve the kernel performance.
    for(i = 0;i < N_ELEMENTS;i++)
    {
        g_odata[0] += g_idata[i];
    }

    __syncthreads(); //Syncthreads forces all threads within a block to reach this point before continuing past. Note this is
    //necessary within blocks because not all threads can physically execute at the same time.
    //Syncthreads does NOT synchronize different blocks (but you should not need to for this project).
}

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

// regression test functionality
extern "C" 
unsigned int compare( const float* reference, const float* data, 
                      const unsigned int len);
extern "C" 
void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char** argv) 
{
    runTest( argc, argv);
    cutilExit(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a scan test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    cudaSetDevice( cutGetMaxGflopsDeviceId() );

    int num_elements = N_ELEMENTS;
    cutGetCmdLineArgumenti( argc, (const char**) argv, "n", &num_elements);

    unsigned int timer;
    cutilCheckError( cutCreateTimer(&timer));
    
    const unsigned int num_threads = 1;
    const unsigned int mem_size = sizeof( float) * num_elements;

    // allocate host memory to store the input data
    float* h_data = (float*) malloc( mem_size);
      
    // initialize the input data on the host to be integer values
    // between 0 and 1000
    printf("INPUT: ");
    for( unsigned int i = 0; i < num_elements; ++i) 
    {
        h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));
        printf(" %f ", h_data[i]);
    }
    printf("\n");

    // compute reference solution
    float* reference = (float*) malloc( mem_size);  
    computeGold( reference, h_data, num_elements);

    // allocate device memory input and output arrays
    float* d_idata;
    float* d_odata;
    cutilSafeCall( cudaMalloc( (void**) &d_idata, mem_size));
    cutilSafeCall( cudaMalloc( (void**) &d_odata, mem_size));

    // copy host memory to device input array
    cutilSafeCall( cudaMemcpy( d_idata, h_data, mem_size, cudaMemcpyHostToDevice) );

    // setup execution parameters
    // Note that these scans only support a single thread-block worth of data,
    // but we invoke them here on many blocks so that we can accurately compare
    // performance
#ifndef __DEVICE_EMULATION__
    dim3  grid(1, 1, 1);  
#else
    dim3  grid(1, 1, 1); // only one run block in device emu mode or it will be too slow
#endif
    dim3  threads(num_threads, 1, 1);

    // make sure there are no CUDA errors before we start
    CUT_CHECK_ERROR("Kernel execution failed");

    printf("Running sum of %d elements\n", num_elements);
  
    // execute the kernels
    unsigned int numIterations = 100;

    cutStartTimer(timer);
    for (int i = 0; i < numIterations; ++i)
    {
        sum_kernel<<< grid, threads >>>
            (d_odata, d_idata, num_elements);
    }
    cudaThreadSynchronize();
    cutStopTimer(timer);
    printf("Average time: %f ms\n\n", cutGetTimerValue(timer) / numIterations);

    cutResetTimer(timer);

    // check for any errors
    cutilCheckMsg("Kernel execution failed");

    // check results
    // copy result from device to host
    cutilSafeCall(cudaMemcpy( h_data, d_odata, sizeof(float) * num_elements, 
                                   cudaMemcpyDeviceToHost));

    printf("OUTPUT: ");
    printf(" %f ", h_data[0]);
    printf("\n");
    printf("REFERENCE: ");
    printf(" %f ", reference[0]);
    printf("\n");

    // custom output handling when no regression test running
    // in this case check if the result is equivalent to the expected soluion
    
    // Due to the large number of additions, a non-zero epsilon is necessary to
    // mask floating point precision errors.
    float epsilon = 0.0f;
    unsigned int result_regtest = cutComparefe( reference, h_data, 1, epsilon);
    printf( "sum: Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");

    // cleanup memory
    free( h_data);
    free( reference);
    cutilSafeCall(cudaFree(d_idata));
    cutilSafeCall(cudaFree(d_odata));
    cutilCheckError(cutDeleteTimer(timer));
}
