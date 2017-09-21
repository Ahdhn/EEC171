/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESqS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* Template project which demonstrates the basics on how to setup a project 
* example application.
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <helper_cuda.h>

#define LOOPS 10000
#define USE_ALL_REGS

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel template for flops test
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel( float* g_idata, float* g_odata, float value) 
{
	// global ID of the thread
	int id = blockIdx.x*blockDim.x + threadIdx.x;
    
    float result=value;
    
    // We have 32 registers here
    // Try adding more to see how performance varies
    float val1 = g_idata[id];
    float val2 = sin(result*result+2);
    float val3 = sin(result*result+5);
    float val4 = sin(result*result+45);
    float val5 = sin(result*result+7892);
    float val6 = sin(result*result+72);
    float val7 = sin(result*result+2);
    float val8 = sin(result*result+2);
    float val9 = sin(result*result+2);
    float val10 = sin(result*result+2);
    float val11 = sin(result*result+2);
    float val12 = sin(result*result+2);
    float val13 = sin(result*result+2);
    float val14 = sin(result*result+2);

    float val15 = sin(result*result+2);
    float val16 = sin(result*result+2);
    float val17 = sin(result*result+2);
    float val18 = sin(result*result+2);
    float val19 = sin(result*result+2);
    float val20 = sin(result*result+2);
    float val21 = sin(result*result+2);
    float val22 = sin(result*result+2);
    float val23 = sin(result*result+2);
    float val24 = sin(result*result+2);
    float val25 = sin(result*result+2);
    float val26 = sin(result*result+2);
    float val27 = sin(result*result+2);
    float val28 = sin(result*result+2);
    float val29 = sin(result*result+2);
    float val30 = sin(result*result+2);
    float val31 = sin(result*result+2);
    float val32 = sin(result*result+2);

    /*float val33 = sin(result*result+33);
    float val34 = sin(result*result+34);
    float val35 = sin(result*result+35);
    float val36 = sin(result*result+36);
    float val37 = sin(result*result+37);
    float val38 = sin(result*result+38);
    float val39 = sin(result*result+39);
    float val40 = sin(result*result+40);
    float val41 = sin(result*result+41);
    float val42 = sin(result*result+42);

    float val43 = sin(result*result+43);
    float val44 = sin(result*result+44);
    float val45 = sin(result*result+45);
    float val46 = sin(result*result+46);
    float val47 = sin(result*result+47);
    float val48 = sin(result*result+48);
    float val49 = sin(result*result+49);
    float val50 = sin(result*result+50);
    float val51 = sin(result*result+51);
    float val52 = sin(result*result+52);

    float val53 = sin(result*result+53);
    float val54 = sin(result*result+54);*/
    
       

	for(int i=0; i<LOOPS; i++)
	{
#ifdef USE_ALL_REGS
		// Uses all of the above registers
		float x = val1;
		//x+=31*val2;
		x+=val2;
		x+=val3;
		x+=val4;
		x+=val5;
		x+=val6;
		x+=val7;
		x+=val8;
		x+=val9;
		x+=val10;
		x+=val11;
		x+=val12;
		x+=val13;
		x+=val14;
		x+=val15;
		x+=val16;
		x+=val17;
		x+=val18;
		x+=val19;
		x+=val20;
		x+=val21;
		x+=val22;
		x+=val23;
		x+=val24;
		x+=val25;
		x+=val26;
		x+=val27;
		x+=val28;
		x+=val29;
		x+=val30;
		x+=val31;
		x+=val32;

		/*x+=val33;
		x+=val34;
		x+=val35;
		x+=val36;
		x+=val37;
		x+=val38;
		x+=val39;
		x+=val40;
		x+=val41;
		x+=val42;

		x+=val43;
		x+=val44;
		x+=val45;
		x+=val46;
		x+=val47;
		x+=val48;
		x+=val49;
		x+=val50;
		x+=val51;
		x+=val52;

		x+=val53;
		x+=val54;*/
		

		result += x;
#else	
		// Uses only a few of the above registers
		// but does the same math
		
		// TODO:
		// 
		// Write a sequence of operations that will perform the same math/computation
		// as above, except it will use less registers.
		// 
		// Hint: Notice that the values of val1,val2,...,val32 are the same.
		// However, the compiler still uses all the registers in the above even though
		// the values are the same. You can do the above with much less registers.
		// Verify your code uses less registers by using the option 'ptxas=1' during compilation.
		//
		// Don't forget to comment out the "#define USE_ALL_REGS" line above.
		float x = val1;
        for(int i=0;i<31;i++){
			x+=val2;
        }
		result += x;
		
#endif

		__syncthreads();
	}

	// write final output
	g_odata[id] = result;
}

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);

    //cutilExit(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{

    cudaSetDevice(1);

    //unsigned int timer = 0;
    // cutCreateTimer( &timer));
    //cutStartTimer( timer);
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    // adjust number of threads & blocks here
    unsigned int num_threads_per_block = 128;
    unsigned int num_blocks = 2048;
    unsigned int num_threads = num_threads_per_block * num_blocks;
    unsigned int mem_size = sizeof(float) * num_threads;
    
    // allocate host memory
    float* h_idata = (float*) malloc(mem_size);
    
    // initalize the memory
    for( unsigned int i = 0; i < num_threads; ++i) 
    {
        h_idata[i] = (float) i;
    }

    // allocate device memory
    float* d_idata;
    checkCudaErrors(cudaMalloc( (void**) &d_idata, mem_size));
    
    // copy host memory to device
    checkCudaErrors(cudaMemcpy( d_idata, h_idata, mem_size,cudaMemcpyHostToDevice));

    // allocate device memory for result
    float* d_odata;
    checkCudaErrors(cudaMalloc( (void**) &d_odata, mem_size));

    // setup execution parameters
    // adjust thread block sizes here
    dim3  grid(num_blocks, 1, 1);
    dim3  threads(num_threads_per_block, 1, 1);

    // execute the kernel
    testKernel<<< grid, threads >>>( d_idata, d_odata, 5);

    // allocate mem for the result on host side
    float* h_odata = (float*) malloc( mem_size);
    
    // copy result from device to host
    checkCudaErrors(cudaMemcpy( h_odata, d_odata, sizeof( float) * num_threads, cudaMemcpyDeviceToHost));

    //cutStopTimer( timer);
    checkCudaErrors(cudaEventRecord(stop));
    float milliseconds = -1;
    cudaDeviceSynchronize();
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
    printf( "h_odata= %f\n", *h_odata);
    printf( "Processing time: %f (ms)\n", milliseconds);
    // cleanup memory
    free( h_idata);
    free( h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);
}
