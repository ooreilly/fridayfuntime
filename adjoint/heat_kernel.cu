#include <stdio.h>
#include <assert.h>

const int radius = 4;

#define swap(x, y) { double t = (x); (x) = (y); (y) = (t); }

__inline__ __device__ void periodic_bc(double *u, const int n) {

        int idx = threadIdx.x + blockDim.x * blockIdx.x;

        if (idx >= radius) return;

        assert(radius - idx - 1 >= 0);
        assert(n - radius + idx < n);
        swap(u[radius - idx - 1], u[n - radius + idx]);
}

__global__ void heat_kernel(double *u, const int n, const double update,
                            const double *a, const double dt, const double h) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;

        double coeff[] =
            {-1.0 / 560, 
              8.0 / 315,
               -1.0 / 5,  
                8.0 / 5,
                -205.0 / 72,
                8.0 / 5,
                -1.0 / 5, 
                8.0 / 315,
                -1.0 / 560};


        for (int i = idx + radius; i < n - radius; i += blockDim.x * gridDim.x) {
                
                double v = 0.0;
                #pragma unroll
                for (int j = 0; j < 2 * radius + 1; ++j)
                        v += a[i] * coeff[j] * u[i + j - radius];


                double alpha = dt / (h * h);
                u[i] = update * u[idx] + alpha * v;

        }

        periodic_bc(u, n);
}


__global__ void test_periodic_bc(double *u, const int n) {
        periodic_bc(u, n);
}
