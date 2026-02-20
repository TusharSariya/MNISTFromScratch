#include <stdio.h>

__global__ void multiply_by_two(int *val) {
    __shared__ int s_val;
    s_val = *val;
    s_val *= 2;
    *val = s_val;
}


__global__ void learn(double *streetlights, double *walkstop, double *weights_0, double *weights_1) {
    __shared__ double streetlights[4][3];
    __shared__ double walkstop[4][1];
    __shared__ double weights_0[4][3]; 
    __shared__ double weights_1[4][1];
}

int main() {
    int host_val = 21;
    int *device_val;
    cudaError_t err;

    err = cudaMalloc(&device_val, sizeof(int));
    if (err != cudaSuccess) { printf("cudaMalloc: %s\n", cudaGetErrorString(err)); return 1; }

    err = cudaMemcpy(device_val, &host_val, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("memcpy H2D: %s\n", cudaGetErrorString(err)); return 1; }

    multiply_by_two<<<1, 1>>>(device_val);

    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("kernel launch: %s\n", cudaGetErrorString(err)); return 1; }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("sync: %s\n", cudaGetErrorString(err)); return 1; }

    err = cudaMemcpy(&host_val, device_val, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("memcpy D2H: %s\n", cudaGetErrorString(err)); return 1; }

    cudaFree(device_val);

    printf("%d\n", host_val);

    double streetlights[4][3] = {
        {1,0,1},
        {0,1,1},
        {0,0,1},
        {1,1,1}
    };

    double walkstop[4][1] = {
        {1},
        {1},
        {0},
        {1}
    };

    double *streetlights_d;
    double *walkstop_d;

    err = cudaMalloc(&streetlights_d,12*sizeof(double));
    if (err != cudaSuccess) { printf("cudaMalloc: %s\n", cudaGetErrorString(err)); return 1; }

    err = cudaMemcpy(streetlights_d, &streetlights, 12*sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("memcpy H2D: %s\n", cudaGetErrorString(err)); return 1; }

    err = cudaMalloc(&walkstop_d,4*sizeof(double));
    if (err != cudaSuccess) { printf("cudaMalloc: %s\n", cudaGetErrorString(err)); return 1; }

    err = cudaMemcpy(walkstop_d, &walkstop, 4*sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("memcpy H2D: %s\n", cudaGetErrorString(err)); return 1; }

    double *weights_0_h = malloc(12 * sizeof(double));

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            double normalized = (double)rand() / (double)RAND_MAX;
            double random_value = (normalized * 2.0) - 1.0;
            weights_0_h[4*i+j] = random_value;
        }
    };


    double *weights_1_h = malloc(4 * sizeof(double));

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 1; j++) {
            double normalized = (double)rand() / (double)RAND_MAX;
            double random_value = (normalized * 2.0) - 1.0;
            weights_1_h[i+j] = random_value;
        }
    };

    double *weights_0_d;
    double *weights_1_d;

    err = cudaMalloc(&weights_0_d,12*sizeof(double));
    if (err != cudaSuccess) { printf("cudaMalloc: %s\n", cudaGetErrorString(err)); return 1; }

    err = cudaMemcpy(weights_0_d, weights_0_h, 12*sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("memcpy H2D: %s\n", cudaGetErrorString(err)); return 1; }

    err = cudaMalloc(&weights_1_d,4*sizeof(double));
    if (err != cudaSuccess) { printf("cudaMalloc: %s\n", cudaGetErrorString(err)); return 1; }
    
    err = cudaMemcpy(weights_1_d, weights_1_h, 4*sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("memcpy H2D: %s\n", cudaGetErrorString(err)); return 1; }



    return 0;
}
