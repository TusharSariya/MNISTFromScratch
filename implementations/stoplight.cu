#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void multiply_by_two(int *val) {
    __shared__ int s_val;
    s_val = *val;
    s_val *= 2;
    *val = s_val;
}
//threadIdx.x   // thread index within its block (0 to blockDim.x - 1)
//blockIdx.x    // which block this thread is in (0 to gridDim.x - 1)
//blockDim.x    // how many threads per block
//gridDim.x     // how many blocks total

__global__ void learn(double *streetlights, double *walkstop, double *weights_0, double *weights_1) {
    __shared__ double s_streetlights[4][3];
    __shared__ double s_walkstop[4];
    __shared__ double s_weights_0[3][4];
    __shared__ double s_weights_1[4];
    __shared__ double s_weights_1_new[4];

    __shared__ double s_output_0[4][4];

    __shared__ double prediction[4][1];

    __shared__ double relu_output_0_T[4][4];

    __shared__ double delta_1[4][1];

    __shared__ double error_1[4][1];

    __shared__ double error_0[3][4];

    __shared__ double delta_0[4][4];

    __shared__ double relu_mask[4][4];

    __shared__ int done;

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 12)
        s_streetlights[i/3][i%3] = streetlights[i];

    if (i < 4)
        s_walkstop[i] = walkstop[i];

    if (i < 12)
        s_weights_0[i/4][i%4] = weights_0[i];

    if (i < 4)
        s_weights_1[i] = weights_1[i];

    __syncthreads();

    int row_0 = i / 4;
    int col_0 = i % 4;

    ///forward pass


    //mat mul 4X3 and 3X4 -> 4X4
    for(int idx = 0; idx < 1000; idx++) {
        double total = 0.0;
        for (int j = 0; j < 3; j++) {
            total += s_streetlights[row_0][j]*s_weights_0[j][col_0];
        }
        s_output_0[row_0][col_0] = total;

        //relu 
        s_output_0[row_0][col_0] = fmax(s_output_0[row_0][col_0],0.0);

        //mat mul 4X4 and 4X1 -> 4X1
        if (i < 4) {
            prediction[i][0] = 0;
            for (int k = 0; k < 4; k++) {
                prediction[i][0] += s_output_0[i][k] * s_weights_1[k];
            };
        }

        __syncthreads(); 
        //error (MSE)
        done = 0;
        double sum = 0.0;
        if (i == 0) {
            for (int l = 0; l < 4; l++) {
                double diff = prediction[l][0] - s_walkstop[l];
                sum += diff * diff;
            }
            double mse = sum / 4.0;
            printf("iter %d, error: %.4f\n", idx + 1, mse);
            if(mse < 0.01 && idx % 20 == 0) {
                done = 1;
            }
        }
        __syncthreads(); 
        if(done) break;


        ///backprop

        //transpose
        relu_output_0_T[col_0][row_0] = s_output_0[row_0][col_0];

        if (i < 4) {
            delta_1[i][0] = prediction[i][0] - s_walkstop[i];
        }

        if (i < 4) {
            //mat mul 4X4 and 4X1 -> 4X1 and scale
            error_1[i][0] = 0;
            for (int k = 0; k < 4; k++) {
                error_1[i][0] += relu_output_0_T[i][k] * delta_1[k][0] * 0.1;
            };
            s_weights_1_new[i] = s_weights_1[i] - error_1[i][0];
        }
        
        // 4X1 and 1X4 (but fake because its not worth it) -> 4X4
        delta_0[row_0][col_0] = delta_1[row_0][0]*s_weights_1[col_0];

        relu_mask[row_0][col_0] = s_output_0[row_0][col_0] > 0 ? 1.0 : 0.0;

        delta_0[row_0][col_0]= delta_0[row_0][col_0]*relu_mask[row_0][col_0];

        __syncthreads(); 

        if (i < 12) {
            error_0[row_0][col_0] = 0;
            for (int j = 0; j < 4; j++) {
                //3X4 (fake the transpose) and 4X4 -> 3X4
                error_0[row_0][col_0] += s_streetlights[j][row_0]*delta_0[j][col_0]*0.1;
            }
            s_weights_0[row_0][col_0] = s_weights_0[row_0][col_0] - error_0[row_0][col_0];
        }
        if (i < 4) {
            s_weights_1[i] = s_weights_1_new[i];
        }

        __syncthreads(); 

    }



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

    //unlike C you have to cast this to double *
    double *weights_0_h = (double*)malloc(12 * sizeof(double));

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            double normalized = (double)rand() / (double)RAND_MAX;
            double random_value = (normalized * 2.0) - 1.0;
            weights_0_h[4*i+j] = random_value;
        }
    };


    double *weights_1_h = (double*)malloc(4 * sizeof(double));

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

    learn<<<1,16>>>(streetlights_d,walkstop_d,weights_0_d,weights_1_d);
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("learn launch: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("learn sync: %s\n", cudaGetErrorString(err)); return 1; }

    free(weights_0_h);
    free(weights_1_h);
    cudaFree(streetlights_d);
    cudaFree(walkstop_d);
    cudaFree(weights_0_d);
    cudaFree(weights_1_d);

    return 0;
}
