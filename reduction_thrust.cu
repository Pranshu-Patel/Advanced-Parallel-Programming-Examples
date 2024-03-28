#include <sys/time.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#define NTESTS 10

#define BLOCK_SIZE 256

#define NMAX 131072

#define NBLOCKS NMAX/BLOCK_SIZE

float h_A[NMAX];


int
timeval_subtract (double *result, struct timeval *x, struct timeval *y)
{
struct timeval result0;
if (x->tv_usec < y->tv_usec) {
int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
y->tv_usec -= 1000000 * nsec;
y->tv_sec += nsec;
}
if (x->tv_usec - y->tv_usec > 1000000) {
int nsec = (y->tv_usec - x->tv_usec) / 1000000;
y->tv_usec += 1000000 * nsec;
y->tv_sec -= nsec;
}
result0.tv_sec = x->tv_sec - y->tv_sec;
result0.tv_usec = x->tv_usec - y->tv_usec;
*result = ((double)result0.tv_usec)/1e6 + (double)result0.tv_sec;
return x->tv_sec < y->tv_sec;
}

int main (int argc,char **argv)
{
struct timeval tdr0, tdr1, tdr;
double restime, min0;
int error;
float *d_A;

cudaMalloc((void **) &d_A, NMAX*sizeof(float));

int kk;
for (kk=0; kk<NTESTS; kk++)
{

for (int i=0; i<NMAX; i++)
{
h_A[i] = (float)rand()/(float)RAND_MAX;
}

min0 = h_A[0];
for (int i=1; i<NMAX; i++)
if (h_A[i] < min0)
min0 = h_A[i];


if (error = cudaMemcpy( d_A, h_A, NMAX*sizeof(float), cudaMemcpyHostToDevice))
{
printf ("Error %d\n", error);
exit (error);
}

gettimeofday (&tdr0, NULL);

thrust::device_ptr<float> d_ptr_A(d_A);
float reduction_min = thrust::reduce(d_ptr_A, d_ptr_A + NMAX, min0, thrust::minimum<float>());

gettimeofday (&tdr1, NULL);
tdr = tdr0;
timeval_subtract (&restime, &tdr1, &tdr);

printf ("Min: %e (relative error %e)\n", reduction_min, fabs((double)reduction_min-min0)/min0);

printf ("Time: %e\n", restime);

}


cudaFree(d_A);
return 0;

}
