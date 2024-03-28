#include <stdio.h>
#include <cuda_runtime.h>

#define ARRAY_SIZE (1024 * 1024)
#define MAX_STRIDE 32

__global__ void accessArrayWithStride(int *arr, int stride, double *bandwidth) {
int idx = threadIdx.x + blockIdx.x * blockDim.x;
int i;
double sum = 0.0;

for (i = 0; i < ARRAY_SIZE / (sizeof(int) * stride); ++i) {
int val = arr[i * stride];
sum += val;
}

bandwidth[idx] = sum;
}

int main() {
int *arr;
double *bandwidth;
cudaEvent_t start, stop;

cudaMallocManaged(&arr, ARRAY_SIZE * sizeof(int));
cudaMallocManaged(&bandwidth, MAX_STRIDE * sizeof(double));
if (arr == NULL || bandwidth == NULL) {
printf("Memory allocation failed.\n");
return 1;
}

for (int i = 0; i < ARRAY_SIZE; ++i) {
arr[i] = i;
}

int blockSize = 256;
int numBlocks = (MAX_STRIDE + blockSize - 1) / blockSize;

printf("Testing memory bandwidth with different strides...\n");
printf("Stride\tBandwidth (MB/s)\n");

for (int stride = 1; stride <= MAX_STRIDE; ++stride) {
printf("Testing stride: %d\n", stride);

cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
accessArrayWithStride<<<numBlocks, blockSize>>>(arr, stride, bandwidth);
cudaEventRecord(stop);
cudaDeviceSynchronize();

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
double bw = (double)(ARRAY_SIZE) / (milliseconds * 1024.0 * 1024.0);
printf("%d\t%.2f\n", stride, bw);

cudaEventDestroy(start);
cudaEventDestroy(stop);
}

cudaFree(arr);
cudaFree(bandwidth);

return 0;
}
