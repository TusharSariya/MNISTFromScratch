#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>


// returns 1.0 where arr > 0, else 0.0 — used in backprop
double *relu_deriv(double *arr, int y, int x) {
    double *out = malloc(y*x*sizeof(double));
    for (int j = 0; j < y; j++) {
        for (int i = 0; i < x; i++) {
            out[j*x+i] = arr[j*x+i] > 0 ? 1.0 : 0.0;
        }
    }
    return out;
}

//can optimize with SIMD
double *relu(double *arr, int y, int x) {
    for (int j = 0; j < y; j++) {
        for (int i = 0; i < x; i++) {
            if (arr[i+j*x] < 0) {
                arr[i+j*x] = 0;
            }
        }
    }
    return arr;
}

double *error(double *pred,double *goal, int y, int x) {
    double *error = malloc(y*x*sizeof(double));
    for (int j = 0; j < y; j++) {
        for (int i = 0; i < x; i++) {
            double temp = pred[j*x+i]-goal[j*x+i];
            error[j*x+i]= temp * temp;
        }
    }
    return error;
}

// left (y,x) right (x,z) out (y,z)
//these are row major, (row,col)
//counterintuitively the stride is the X axis
double *matmul(double* left, double* right,int y, int x, int z) {
    double *out = malloc(y*z*sizeof(double)); 
    for (int j = 0; j < y; j++) {
        for (int i = 0; i < z; i++) {
            out[j*z+i] = 0;
            for (int k = 0; k < x; k++) {
                //dot product is left row * right col and iterating respectively
                out[j*z+i] += left[j*x+k]*right[z*k+i];
            }
        }
    }
    return out;
}

// scale each element by a scalar factor
double *scale(double *arr, int y, int x, double factor) {
    double *out = malloc(y*x*sizeof(double));
    for (int j = 0; j < y; j++) {
        for (int i = 0; i < x; i++) {
            out[j*x+i] = arr[j*x+i] * factor;
        }
    }
    return out;
}

// in (y,x) out (x,y)
double *transpose(double *in, int y, int x) {
    double *out = malloc(y*x*sizeof(double));
    for (int j = 0; j < y; j++) {
        for (int i = 0; i < x; i++) {
            out[i*y+j] = in[j*x+i];
        }
    }
    return out;
}



// pred (y,x) goal (y,x) out (y,x)
double *substract(double* pred, double* goal, int y, int x) {
    double *out = malloc(y*x*sizeof(double));
    for (int j = 0; j < y; j++) {
        for (int i = 0; i < x; i++) {
            out[j*x+i] = pred[j*x+i] - goal[j*x+i];
        }
    }
    return out;
}

// element-wise multiply, a (y,x) b (y,x) out (y,x)
double *multiply(double* a, double* b, int y, int x) {
    double *out = malloc(y*x*sizeof(double));
    for (int j = 0; j < y; j++) {
        for (int i = 0; i < x; i++) {
            out[j*x+i] = a[j*x+i] * b[j*x+i];
        }
    }
    return out;
}


int main() {
    printf("Hello, World!\n");
    srand(time(NULL));

    //the c compiler will allocate memory contigiously
    double streetlights[4][3] = {
        {1,0,1},
        {0,1,1},
        {0,0,1},
        {1,1,1}
    };

    double *streetlights_h = malloc(12 * sizeof(double));

    //uses SIMD
    //although i can go straight to heap it is much easier to read
    memcpy(streetlights_h,streetlights,12*sizeof(double));

    double walkstop[4][1] = {
        {1},
        {1},
        {0},
        {1}
    };

    double *walkstop_h = malloc(4*sizeof(double));

    memcpy(walkstop_h,walkstop,4*sizeof(double));

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

    int idx = 0;

    for(idx; idx < 1000; idx++) {
        // 4X3 and 3X4 -> 4X4
        double* output_0 = matmul(streetlights_h,weights_0_h,4,3,4);

        //4X4
        double* relu_output_0 = relu(output_0,4,4);

        // 4X4 and 4X1 -> 4X1
        double* pred = matmul(relu_output_0,weights_1_h,4,4,1);

        double* err = error(pred,walkstop_h,4,1);

        double sum = 0;
        for (int i = 0; i < 4; i++)
            sum += err[i];
        printf("mean error: %f\n", sum / 4.0);
        if(sum/4.0 < 0.01 && idx % 20 == 0) {
            break;
        }

        // backprop layer 1
        double *relu_output_0_T = transpose(relu_output_0, 4, 4);
        double *delta_1 = substract(pred, walkstop_h, 4, 1);
        double *error_1 = matmul(relu_output_0_T, delta_1, 4, 4, 1);
        double *error_1_scaled = scale(error_1, 4, 1, 0.1);
        double *weight_1_new = substract(weights_1_h, error_1_scaled, 4, 1);

        // backprop layer 0
        double *weights_1_h_T = transpose(weights_1_h, 4, 1);
        double *delta_0 = matmul(delta_1, weights_1_h_T, 4, 1, 4);
        // mask delta_0 by where the forward pass activations were > 0
        double *relu_mask = relu_deriv(relu_output_0, 4, 4);
        double *delta_0_relu = multiply(delta_0, relu_mask, 4, 4);
        free(relu_mask);
        double *streetlights_h_T = transpose(streetlights_h, 4, 3);
        double *error_0 = matmul(streetlights_h_T, delta_0_relu, 3, 4, 4);
        double *error_0_scaled = scale(error_0, 3, 4, 0.1);
        double *weight_0_new = substract(weights_0_h, error_0_scaled, 3, 4);

        free(output_0);
        free(pred);
        free(err);
        free(relu_output_0_T);
        free(delta_1);
        free(error_1);
        free(error_1_scaled);
        free(weights_1_h_T);
        free(delta_0);
        free(delta_0_relu);
        free(streetlights_h_T);
        free(error_0);
        free(error_0_scaled);

        free(weights_0_h);
        weights_0_h = weight_0_new;

        free(weights_1_h);
        weights_1_h = weight_1_new;
    }

    

    
    return 0;
}
