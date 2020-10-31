#include <stdio.h>
#include <assert.h>

const int radius = 4;

#define swap(x, y) { double t = (x); (x) = (y); (y) = (t); }

__inline__ __device__ void periodic_bc(double *u, const int n) {

        int idx = threadIdx.x + blockDim.x * blockIdx.x;

        if (idx >= radius) return;

        // Copy the left data points [u4, u5, u6, u7]  to the right buffer 
        // 0    1    2    3     4    5    6    7    8    9    10   11   12   13    14   15   16  17
        // o----o----o----o--|--o----o----o----o----o----o----o----o----o----o--|--o----o----o----o
        //u0   u1   u2   u3    u4   u5                                            u4   u5   u6   u7
        u[n - radius + idx] = u[radius + idx];
        //
        // Copy the right data points [u10, u11, u12, u13] to the left buffer
        // 0    1    2    3     4    5    6    7    8    9    10   11   12   13    14   15   16  17
        // o----o----o----o--|--o----o----o----o----o----o----o----o----o----o--|--o----o----o----o
        //u10  u11  u12  u13   u4   u5                        u10  u11  u12  u13                          
        u[idx] = u[n - 2 * radius + idx];
}

__global__ void heat_kernel(double *v, const double *u, const int n, const double update,
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
                
                double D2u = 0.0;
                #pragma unroll
                for (int j = 0; j < 2 * radius + 1; ++j)
                        D2u += coeff[j] * u[i + j - radius];


                double alpha = dt / (h * h);
                v[i] = update * u[i] + alpha * a[i] * D2u;

        }

        periodic_bc(v, n);
}


__global__ void test_periodic_bc(double *u, const int n) {
        periodic_bc(u, n);
}
