#include "stdio.h"
#include "cublas_v2.h"

#define M 5
#define N 3
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

void printMat(float* mat, int m, int n){
    for(int row = 0; row < m; row++){
        for(int col = 0; col < n; col++){
            printf("%.f ",mat[IDX2C(row, col, m)]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void){
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    int row, col;
    float* h_A; // 5x3 vector
    float* h_X; // 3x1 matrix
    float* h_Y; // 5x1 vector
    float* d_A = 0;
    float* d_X = 0;
    float* d_Y = 0;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Allocating host memory
    h_A = (float*)malloc(M*N*sizeof(*h_A));
    h_X = (float*)malloc(N*1*sizeof(*h_X));
    h_Y = (float*)malloc(M*1*sizeof(*h_Y));
    if (!h_A || !h_X || !h_Y){
        printf("host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initilize host memory 
    for (col = 0; col < N; col++){
        for(row = 0; row < M; row++){
            h_A[IDX2C(row, col, M)] = (float)(row*N + col + 1);
        }
    }
    for (row = 0 ; row < N ; row++){
        h_X[row] = (float)(row+1);
    }
    // printMat(h_X, N, 1);
    // for (int i = 0; i < M*N ; i++){
    //     printf("%.f ", h_A[i]);
    // }
    // printf("\n");
    // printf("host A(%dx%d) = \n", M, N);
    // printMat(h_A, M, N);

    // Allocating device memory
    cudaStat = cudaMalloc((void**)&d_A, M*N*sizeof(float));
    cudaStat = cudaMalloc((void**)&d_X, N*1*sizeof(float));
    cudaStat = cudaMalloc((void**)&d_Y, M*1*sizeof(float));
    if (cudaStat != cudaSuccess){
        printf("device memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize CUBLAS & Create Handle
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS){
        printf("CUBLAS initilization failed\n");
        return EXIT_FAILURE;
    }

    // Copying matrix values to device memory
    stat = cublasSetMatrix(M, N, sizeof(float), h_A, M, d_A, M);
    // stat = cublasSetMatrix(N, 1, sizeof(float), h_X, N, d_X, N);
    // stat = cublasSetMatrix(M, 1, sizeof(float), h_Y, M, d_Y, M);
    stat = cublasSetVector(N, sizeof(float), h_X, 1, d_X, 1);
    stat = cublasSetVector(M, sizeof(float), h_Y, 1, d_Y, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed\n");
        cudaFree(d_A);
        cudaFree(d_X);
        cudaFree(d_Y);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    // Run cublas 32bit integer type gemm
    stat = cublasSgemv(handle, CUBLAS_OP_N, M, N, &alpha, d_A, M, d_X, 1, &beta, d_Y, 1);
    printf("stat: %d\n", stat);
    if(stat != CUBLAS_STATUS_SUCCESS) {
        printf ("cublas gemv error\n");
        cudaFree(d_A);
        cudaFree(d_X);
        cudaFree(d_Y);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    // copy result from device to host
    stat = cublasGetVector(M, sizeof(float), d_Y, 1, h_Y, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree(d_A);
        cudaFree(d_X);
        cudaFree(d_Y);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    cudaFree(d_A);
    cudaFree(d_X);
    cudaFree(d_Y);

    cublasDestroy(handle);

    // Print ouput
    printf("host Y(%dx%d) = \n", M, 1);
    printMat(h_Y, M, 1);

    free(h_A);
    free(h_X);
    free(h_Y);

    return EXIT_SUCCESS;
}
