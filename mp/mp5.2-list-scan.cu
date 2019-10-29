// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512
#define SECTION_SIZE BLOCK_SIZE * 2

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *X, float *Y, float *S, int inputSize, int const section_size) {
  // Use Brent-Kung algorithm

  // CAUTION! the length parameter of shared memory should be a constant,
  // SECTION_SIZE should be bigger than section_size
  __shared__ float XY[SECTION_SIZE];

  int i = 2*blockIdx.x*blockDim.x + threadIdx.x;
  if (i < inputSize) XY[threadIdx.x] = X[i];
  if (i+blockDim.x < inputSize) XY[threadIdx.x+blockDim.x] = X[i+blockDim.x];

  // Reduection
  for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
    __syncthreads();
    int index = (threadIdx.x+1) * 2* stride -1;
    if (index < section_size) {
      XY[index] += XY[index - stride];
    }
  }

  // Distribution
  for (int stride = ceil(section_size/4.0); stride > 0; stride /= 2) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index + stride < section_size) {
      XY[index + stride] += XY[index];
    }
  }

  __syncthreads();
  if (i < inputSize) Y[i] = XY[threadIdx.x];
  if (i+blockDim.x < inputSize) Y[i+blockDim.x] = XY[threadIdx.x+blockDim.x];


  if (S) {
    __syncthreads();
    if (threadIdx.x == (blockDim.x-1)) {
      S[blockIdx.x] = XY[section_size-1];
    }
  }
}

__global__ void sumUp(float *S, float *Y, int len) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < len && blockIdx.x > 0) {
    Y[i] += S[blockIdx.x-1];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // Define some vars
  int SECTION_CNT = ceil(numElements/(SECTION_SIZE)*1.0);
  float *auxiliary;
  cudaMalloc((void **) &auxiliary, SECTION_CNT * sizeof(float));

  wbTime_start(Compute, "Performing CUDA computation");
  // Phase 1
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  dim3 DimGrid(ceil(numElements/(SECTION_SIZE*1.0)), 1, 1);
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, auxiliary, numElements, SECTION_SIZE);
  cudaDeviceSynchronize();
  
  // Phase 2
  dim3 DimBlock2(ceil(SECTION_CNT/2.0), 1, 1);
  dim3 DimGrid2(1, 1, 1);
  scan<<<DimGrid2, DimBlock2>>>(auxiliary, auxiliary, NULL, SECTION_CNT, SECTION_CNT);
  cudaDeviceSynchronize();

  // Phase 3
  dim3 DimBlock3(SECTION_SIZE, 1, 1);
  dim3 DimGrid3(ceil(numElements/(SECTION_SIZE*1.0)), 1, 1);
  sumUp<<<DimGrid3, DimBlock3>>>(auxiliary, deviceOutput, numElements);
  cudaDeviceSynchronize();

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

