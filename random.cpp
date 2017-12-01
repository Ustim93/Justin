#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ 
void Random(int *c)
{
  unsigned int ind = blockIdx.x*blockDim.x+threadIdx.x;
  curandState_t state;
  curand_init(ind, /* the seed controls the sequence of random values that are produced */
              0, /* the sequence number is only important with multiple cores */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &state);
  c[ind] = 1 + curand(&state)%100;
}

int main(void)
{
  int N = 1<<6;
  int *y, *d_y;
  y = (int*)malloc(N*sizeof(int));

  cudaMalloc(&d_y, N*sizeof(int));

  for (int i = 0; i < N; i++) {
    y[i] = 0;
  }
  cudaMemcpy(d_y, y, N*sizeof(int), cudaMemcpyHostToDevice);

   Random<<<(N+255)/256, 256>>>(d_y);

  cudaMemcpy(y, d_y, N*sizeof(int), cudaMemcpyDeviceToHost);

  //int maxError = 0;
  for (int i = 0; i < N; i++)
  {
   // maxError = max(maxError, abs(y[i]-4));
   printf("Rand is: %d\n", y[i]);
  }
  //printf("Max error: %d  %d\n", maxError, N);

  cudaFree(d_y);
  free(y);
}
