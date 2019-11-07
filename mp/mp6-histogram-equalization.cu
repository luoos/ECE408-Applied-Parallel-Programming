// Histogram Equalization

/* Steps:
1.  Cast the image from float to uint8_t
2.  Convert the image from RGB to GrayScale
3.  Compute the histogram of grayImage
4.  Compute the Cumulative Distribution Function of histogram
5.  Compute the minimum value of the CDF.
    The maximal value of the CDF should be 1.0
6.  Define the histogram equalization function
7.  Apply the histogram equalization function
8.  Cast back to float
*/

#include <wb.h>

typedef unsigned char uint8_t;

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 512

// Kernel 1: float -> unsigned char
__global__ void cast(float *input, uint8_t *output, int len) {
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx < len) {
    output[idx] = (uint8_t) (255 * input[idx]);
  }
}

// Kernel 2: rgb -> grayscale
__global__ void convert(uint8_t *input, uint8_t *output, int len) {
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx < len) {
    uint8_t r = input[3*idx];
    uint8_t g = input[3*idx+1];
    uint8_t b = input[3*idx+2];
    output[idx] = (uint8_t) (0.21*r + 0.71*g + 0.07*b);
  }
}

// Kernel 3: grayscale -> histogram
__global__ void computeHistogram(uint8_t *input, int *output, int len) {
  // blockDim.x should >= 256
  int tid = threadIdx.x;
  int idx = tid + blockDim.x * blockIdx.x;
  __shared__ unsigned int histo_s[HISTOGRAM_LENGTH];

  if (tid < HISTOGRAM_LENGTH) {
    histo_s[tid] = 0;
  }
  __syncthreads();

  if (idx < len) {
    int pos = input[idx];
    atomicAdd(&(histo_s[pos]), 1);
  }
  __syncthreads();

  if (tid < HISTOGRAM_LENGTH) {
    atomicAdd(&(output[tid]), histo_s[tid]);
  }
}

// Kernel 4: histogram -> cdf
// Assume the len is fixed - 256.
__global__ void scan(int *histogram, float *cdf, int len, int pixelLs) {
  __shared__ float XY[HISTOGRAM_LENGTH];

  int i = threadIdx.x;
  if (i < len) XY[i] = histogram[i];
  if (i + blockDim.x < len) XY[i+blockDim.x] = histogram[i+blockDim.x];

  for (int stride = 1; stride <= blockDim.x; stride *= 2) {
    __syncthreads();
    int index = (i+1) * 2 * stride - 1;
    if (index < len) {
      XY[index] += XY[index - stride];
    }
  }

  for (int stride = ceil(len/4.0); stride > 0; stride /= 2) {
    __syncthreads();
    int index = (i+1)*stride*2 - 1;
    if(index + stride < len) {
      XY[index + stride] += XY[index];
    }
  }
  __syncthreads();
  if (i < len) cdf[i] = ((float) (XY[i]*1.0)/pixelLs);
  if (i + blockDim.x < len) cdf[i+blockDim.x] = ((float) (XY[i+blockDim.x]*1.0)/pixelLs);
}

// Kernel 5
__global__ void correctColor(uint8_t *ucharImage, float *cdf, int len) {
  __shared__ float scdf[HISTOGRAM_LENGTH];
  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  if (threadIdx.x < HISTOGRAM_LENGTH) {
    scdf[threadIdx.x] = cdf[threadIdx.x];
  }
  __syncthreads();

  if (idx < len) {
    uint8_t val = ucharImage[idx];
    float tmp = 255 * (scdf[val] - scdf[0]) / (1.0 - scdf[0]);
    ucharImage[idx] = (uint8_t) (min(max(tmp, 0.0), 255.0));
  }
}

// Kernel 6: unsigned char -> float
__global__ void castBack(uint8_t *ucharImage, float *outputImage, int len) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < len) {
    outputImage[idx] = (float) (ucharImage[idx]/255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t     args;
  int         imageWidth;
  int         imageHeight;
  int         imageChannels;
  wbImage_t   inputImage;
  wbImage_t   outputImage;
  float       *hostInputImageData;
  float       *hostOutputImageData;
  float       *deviceImageData;
  uint8_t     *deviceGrayImage;
  int         *deviceHistogram;
  float       *devicecdf;
  uint8_t     *deviceUncharImage;
  const char  *inputImageFile;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage          = wbImport(inputImageFile);
  imageWidth          = wbImage_getWidth(inputImage);
  imageHeight         = wbImage_getHeight(inputImage);
  imageChannels       = wbImage_getChannels(inputImage);
  hostInputImageData  = wbImage_getData(inputImage);
  outputImage         = wbImage_new(imageWidth, imageHeight, imageChannels);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  int imagePixelLs = imageWidth * imageHeight;
  int imageTotalSize = imagePixelLs * imageChannels;
  hostOutputImageData = (float *)malloc(imageTotalSize * sizeof(float));

  cudaMalloc((void **)&deviceImageData,       imageTotalSize    * sizeof(float));
  cudaMalloc((void **)&deviceUncharImage,     imageTotalSize    * sizeof(uint8_t));
  cudaMalloc((void **)&deviceGrayImage,       imagePixelLs      * sizeof(uint8_t));
  cudaMalloc((void **)&deviceHistogram,       HISTOGRAM_LENGTH  * sizeof(int));
  cudaMalloc((void **)&devicecdf,             HISTOGRAM_LENGTH  * sizeof(float));

  cudaMemcpy(deviceImageData, hostInputImageData, imageTotalSize * sizeof(float), cudaMemcpyHostToDevice);

  // Kernel 1: cast
  dim3 DimBlock1(BLOCK_SIZE, 1, 1);
  dim3 DimGrid1(ceil(imageTotalSize/(1.0*BLOCK_SIZE)), 1, 1);
  cast<<<DimGrid1, DimBlock1>>>(deviceImageData, deviceUncharImage, imageTotalSize);
  cudaDeviceSynchronize();

  // Kernel 2: convert
  dim3 DimBlock2(BLOCK_SIZE, 1, 1);
  dim3 DimGrid2(ceil(imagePixelLs/(1.0*BLOCK_SIZE)), 1, 1);
  convert<<<DimGrid2, DimBlock2>>>(deviceUncharImage, deviceGrayImage, imagePixelLs);
  cudaDeviceSynchronize();

  // Kernel 3: compute the histogram
  dim3 DimBlock3(256, 1, 1);
  dim3 DimGrid3(ceil(imagePixelLs/256.0), 1, 1);
  computeHistogram<<<DimGrid3, DimBlock3>>>(deviceGrayImage, deviceHistogram, imagePixelLs);
  cudaDeviceSynchronize();

  // Kernel 4: scan
  dim3 DimBlock4(128, 1, 1);
  dim3 DimGrid4(1, 1, 1);
  scan<<<DimGrid4, DimBlock4>>>(deviceHistogram, devicecdf, 256, imagePixelLs);
  cudaDeviceSynchronize();

  // Kernel 5: correct color
  dim3 DimBlock5(BLOCK_SIZE, 1, 1);
  dim3 DimGrid5(ceil(imageTotalSize/(1.0*BLOCK_SIZE)), 1, 1);
  correctColor<<<DimGrid5, DimBlock5>>>(deviceUncharImage, devicecdf, imageTotalSize);
  cudaDeviceSynchronize();

  // Kernel 6: cast back
  dim3 DimBlock6(BLOCK_SIZE, 1, 1);
  dim3 DimGrid6(ceil(imageTotalSize/(1.0*BLOCK_SIZE)), 1, 1);
  castBack<<<DimGrid6, DimBlock6>>>(deviceUncharImage, deviceImageData, imageTotalSize);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceImageData,
             imageTotalSize * sizeof(float), cudaMemcpyDeviceToHost);

  wbImage_setData(outputImage, hostOutputImageData);
  wbSolution(args, outputImage);

  cudaFree(deviceUncharImage);
  cudaFree(devicecdf);
  cudaFree(deviceGrayImage);
  cudaFree(deviceHistogram);
  cudaFree(deviceImageData);
  free(hostInputImageData);
  free(hostOutputImageData);

  return 0;
}
