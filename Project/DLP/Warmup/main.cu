
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel_OneThread(int N, float*dev_x, float*dev_y)
{    
	for (int i = 0; i < N; i++){
		dev_y[i] = dev_x[i] + dev_y[i];
	}
}

__global__ void addKernel_OneBlock(int N, float*dev_x, float*dev_y)
{
	int id = threadIdx.x;
	int stride = blockDim.x;

	for (int i = id; i < N; i += stride){ 
		dev_y[i] = dev_x[i] + dev_y[i];
	}
}

__global__ void addKernel_ManyBlock(int N, float*dev_x, float*dev_y)
{
	int id = blockIdx.x *blockDim.x + threadIdx.x;
	int stride = blockDim.x*gridDim.x;	

	for (int i = id; i < N; i += stride){
		dev_y[i] = dev_x[i] + dev_y[i];
	}
}

int main()
{
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	int dev = 0;
	cudaSetDevice(dev);
	cudaFree(0);

	cudaError_t error;
	cudaDeviceProp devProp;
	error = cudaGetDeviceProperties(&devProp, dev);

	if (error == cudaSuccess){
		printf("Total number of device: %d", deviceCount);
		printf("\nUsing device Number: %d", dev);
		printf("\n  Device name: %s", devProp.name);
		printf("\n  Compute Capability: v%d.%d", (int)devProp.major, (int)devProp.minor);
		printf("\n  Memory Clock Rate: %d(kHz)", devProp.memoryClockRate);
		printf("\n  Memory Bus Width: %d(bits)", devProp.memoryBusWidth);
		printf("\n  Peak Memory Bandwidth: %f(GB/s)",
			2.0 * devProp.memoryClockRate*(devProp.memoryBusWidth / 8.0) / 1.0E6);		
	}
	else{
		exit(EXIT_FAILURE);
	}

		
	int N = 1 << 20;//1M 

	float *x = new float[N];
	float *y = new float[N];
	
	float *dev_x, *dev_y;	

	//populate x and y
	for (int i = 0; i < N; i++){
		x[i] = 1.0f;
		y[i] = 2.0f;
	}
		
	
	// Allocate GPU buffers for three vectors (two input, one output)
	cudaStatus = cudaMalloc((void**)&dev_x, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaMalloc((void**)&dev_y, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		exit(EXIT_FAILURE);
	}

	// Copy input vectors from host memory to GPU buffers
	cudaStatus = cudaMemcpy(dev_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		exit(EXIT_FAILURE);
	}

	cudaStatus = cudaMemcpy(dev_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		exit(EXIT_FAILURE);
	}


	// Launch a kernel on the GPU with one thread for each element.
	addKernel_OneThread << <1, 1 >> >(N, dev_x, dev_y);
	//addKernel_OneBlock << <1, 256 >> >(N, dev_x, dev_y);
	
	//int blockSize = 256;
	//int numBlocks = (N + blockSize - 1) / blockSize;
	//addKernel_ManyBlock << <numBlocks, blockSize >> >(N, dev_x, dev_y);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}
       
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(y, dev_y, N * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		exit(EXIT_FAILURE);
	}

    //verify 
	float maxErr = 0.0f;
	for (int i = 0; i < N; i++){
		maxErr = std::fmax(maxErr, fabs(y[i] - 3.0f));
	}

	printf("\nMax Error= %f\n", maxErr);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
		exit(EXIT_FAILURE);
    }

	cudaFree(dev_x);
	cudaFree(dev_y);
	free(x);
	free(y);
	
	//system("pause");

    return 0;
}


