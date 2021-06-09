#include "stdio.h"
#include "cublas_v2.h"

#define M 5
#define N 2
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
    float* h_A;
    float* h_B;
    float* h_C;
    float* d_A = 0;
    float* d_B = 0;
    float* d_C = 0;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Allocating host memory
    h_A = (float*)malloc(M * N * sizeof(float));
    h_B = (float*)malloc(N * M * sizeof(float));
    h_C = (float*)malloc(M * M * sizeof(float));
    if (!h_A || !h_B || !h_C){
        printf("host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initilize host memory
    for (col = 0; col < N; col++){
        for(row = 0; row < M; row++){
            h_A[IDX2C(row, col, M)] = (float)(row*N + col + 1);
            h_B[IDX2C(col, row, N)] = (float)(col*M + row + 1);
        }
    }
    for (int i = 0; i < M*N ; i++){
        printf("%.f ", h_A[i]);
    }
    printf("\n");
    for (int i = 0; i < M*N ; i++){
        printf("%.f ", h_B[i]);
    }
    printf("\n");
    printf("host A(%dx%d) = \n", M, N);
    printMat(h_A, M, N);
    printf("host B(%dx%d) = \n", N, M);
    printMat(h_B, N, M);

    for (col = 0; col < M; col++){
        for(row = 0; row < M; row++){
            h_C[IDX2C(row, col, M)] = (float)(row*M + col + 1);
        }
    }
    // printf("host C(%dx%d) = \n", M, M);
    // printMat(h_C, M, M);

    // Allocating device memory
    cudaStat = cudaMalloc((void**)&d_A, M*N*sizeof(float));
    cudaStat = cudaMalloc((void**)&d_B, N*M*sizeof(float));
    cudaStat = cudaMalloc((void**)&d_C, M*M*sizeof(float));
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
    stat = cublasSetMatrix(N, M, sizeof(float), h_B, N, d_B, N);
    stat = cublasSetMatrix(M, M, sizeof(float), h_C, M, d_C, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    // Run cublas 32bit integer type gemm
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, M, N, &alpha, d_A, M, d_B, N, &beta, d_C, M);
    printf("%d",stat);
    
    if (stat == CUBLAS_STATUS_SUCCESS) {
        printf ("cublas gemm error\n");
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    // copy result from device to host
    stat = cublasGetMatrix(M, M, sizeof(float), d_C, M, h_C, M);

   
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cublasDestroy(handle);

    // Print ouput
    printf("host C(%dx%d) = \n", M, M);
    printMat(h_C, M, M);

    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
