#include "stdio.h"
#include "cublas.h"

#define N 10 // 1000*1000 행렬

int main(int argc, char** argv){

    float* h_A;
    float* h_B;
    float* h_C;
    float* d_A = 0;
    float* d_B = 0;
    float* d_C = 0;

    float alpha = 1.0f;
    float beta = 0.0f;
    int n2 = N * N;
    int i = 0;
    int j = 0;

    // Initialize CUBLAS
    cublasInit();

    // Allocating host memory
    h_A = (float*)malloc(n2*sizeof(float));
    h_B = (float*)malloc(n2*sizeof(float));
    h_C = (float*)malloc(n2*sizeof(float));

    // Initilize host memory with random value
    // for (i=0; i < n2; i++){
    //     // h_A[i] = rand() / (float)RAND_MAX;
    //     // h_B[i] = rand() / (float)RAND_MAX;
    //     // h_C[i] = rand() / (float)RAND_MAX;
    //     h_A[i] = 2;
    //     h_B[i] = 2;
    //     h_C[i] = 1;
    // } 
    for (i=0; i < n2; i++){
        h_A[i] = 2;
        h_B[i] = 2;
        h_C[i] = 1;
    } 

    // Allocating device memory
    //// cublasStatus example
    cublasStatus status = cublasAlloc(n2, sizeof(float), (void**)&d_A);
    if(status != CUBLAS_STATUS_SUCCESS){
        fprintf(stderr, "디바이스 메모리 할당 실패\n");
        return EXIT_FAILURE;
    }
    // cublasAlloc(n2, sizeof(float), (void**)&d_A);
    cublasAlloc(n2, sizeof(float), (void**)&d_B);
    cublasAlloc(n2, sizeof(float), (void**)&d_C);

    // Copying matrix values to device memory
    cublasSetVector(n2, sizeof(float), h_A, 1, d_A, 1);
    cublasSetVector(n2, sizeof(float), h_B, 1, d_B, 1);
    cublasSetVector(n2, sizeof(float), h_C, 1, d_C, 1);

    // Run cublas 32bit integer type gemm
    cublasSgemm('n', 'n', N, N, N, alpha, d_A, N, d_B, N, beta, d_C, N);

    // Print ouput
    cublasGetVector(n2, sizeof(float), d_C, 1, h_C, 1);

    // for(i = 1; i <= N; i++){
    //     for(j = 0; j <= N; j++){
    //         printf("%.f ", h_C[j + (i-1)*N]);
    //     }
    //     printf("\n");
    // }

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);

    cublasFree(d_A);
    cublasFree(d_B);
    cublasFree(d_C);

    // Exit cublas
    cublasShutdown();

    return true;
}
