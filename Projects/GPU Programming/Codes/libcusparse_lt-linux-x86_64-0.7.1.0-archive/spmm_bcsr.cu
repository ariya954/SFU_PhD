#include <cuda_runtime.h>
#include <cusparseLt.h>
#include <iostream>
#include <vector>

const char* cusparseLtGetErrorName(cusparseStatus_t status) {
    switch (status) {
        case CUSPARSE_STATUS_SUCCESS: return "CUSPARSE_STATUS_SUCCESS";
        case CUSPARSE_STATUS_NOT_INITIALIZED: return "CUSPARSE_STATUS_NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED: return "CUSPARSE_STATUS_ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE: return "CUSPARSE_STATUS_INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH: return "CUSPARSE_STATUS_ARCH_MISMATCH";
        case CUSPARSE_STATUS_MAPPING_ERROR: return "CUSPARSE_STATUS_MAPPING_ERROR";
        case CUSPARSE_STATUS_EXECUTION_FAILED: return "CUSPARSE_STATUS_EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR: return "CUSPARSE_STATUS_INTERNAL_ERROR";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        default: return "CUSPARSE_STATUS_UNKNOWN";
    }
}

const char* cusparseLtGetErrorString(cusparseStatus_t status) {
    return cusparseLtGetErrorName(status);  // For simplicity, same as name
}

#define CHECK_CUDA(func)                                                        \
{                                                                               \
    cudaError_t status = (func);                                                \
    if (status != cudaSuccess) {                                                \
        std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl;\
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
}

#define CHECK_CUSPARSE(func)                                                    \
{                                                                               \
    cusparseStatus_t status = (func);                                           \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                    \
        std::cerr << "cuSPARSELt error: " << cusparseLtGetErrorName(status)    \
                  << " - " << cusparseLtGetErrorString(status) << std::endl;   \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
}

int main() {
    // Dimensions
    int m = 64;  // rows of A and C
    int k = 64;  // cols of A and rows of B
    int n = 64;  // cols of B and C

    // Block sizes
    int block_size = 4;

    float alpha = 1.0f;
    float beta = 0.0f;

    // Example BSR block pattern (4 nonzero blocks)
    std::vector<int> bsr_row_ptr = {0, 2, 4};         // 2 block rows, each has 2 blocks
    std::vector<int> bsr_col_ind = {0, 2, 1, 3};      // Column indices for 4 blocks

    std::vector<float> bsr_val = {
        // Block (0,0)
          1,0,0,0,
          0,1,0,0,
          0,0,1,0,
          0,0,0,1,
          // Block (0,1)
          2,0,0,0,
          0,2,0,0,
          0,0,2,0,
          0,0,0,2,
        // Block (1,0)
          3,0,0,0,
          0,3,0,0,
          0,0,3,0,
          0,0,0,3,
        // Block (1,1)
          4,0,0,0,
          0,4,0,0,
          0,0,4,0,
          0,0,0,4
        };

    // Dense matrix B
    std::vector<float> hB(k * n, 1.0f); // B is k x n, initialized to ones

    // Allocate GPU memory and copy data from CPU to GPU
    float *dA, *dB, *dC;
    int *d_row_ptr, *d_col_ind;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(float) * bsr_val.size()));
    CHECK_CUDA(cudaMalloc(&d_row_ptr, sizeof(int) * bsr_row_ptr.size()));
    CHECK_CUDA(cudaMalloc(&d_col_ind, sizeof(int) * bsr_col_ind.size()));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(float) * k * n));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * m * n));
    CHECK_CUDA(cudaMemcpy(dA, bsr_val.data(), sizeof(float) * bsr_val.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_row_ptr, bsr_row_ptr.data(), sizeof(int) * bsr_row_ptr.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_ind, bsr_col_ind.data(), sizeof(int) * bsr_col_ind.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeof(float) * k * n, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, sizeof(float) * m * n));

    // cuSPARSELt handle
    cusparseLtHandle_t handle;
    CHECK_CUSPARSE(cusparseLtInit(&handle));

    // Matrix descriptors
    cusparseLtMatDescriptor_t matA, matB, matC;
    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(&handle, &matA, m, k, m, 16, CUDA_R_32F, CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT));
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matB, k, n, k, 16, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle, &matC, m, n, m, 16, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // Matmul descriptor
    cusparseLtMatmulDescriptor_t matmul;
    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&handle, &matmul,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &matA, &matB, &matC, &matC,
                                                  CUSPARSE_COMPUTE_32F));

    // Algorithm selection
    cusparseLtMatmulAlgSelection_t algSel;
    CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle, &algSel, &matmul,
                                                    CUSPARSELT_MATMUL_ALG_DEFAULT));

    // Plan
    cusparseLtMatmulPlan_t plan;
    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &algSel));

    // Workspace
    size_t workspaceSize = 0;
    CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan, &workspaceSize));

    void* dWorkspace = nullptr;
    if (workspaceSize > 0)
        CHECK_CUDA(cudaMalloc(&dWorkspace, workspaceSize));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Launch SpMM
    CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan, &alpha, dA, dB, &beta, dC, dC, dWorkspace, nullptr, 0));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    std::cout << "SpMM execution time: " << elapsed_ms << " ms" << std::endl;

    std::cout << "SpMM using cuSPARSELt completed successfully." << std::endl;

    // Allocate host memory and copy result from device
    float* h_C = (float*) malloc(m * n * sizeof(float));
    cudaMemcpy(h_C, dC, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result matrix C:" << std::endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << h_C[i*n + j] << "\t";
        }
        std::cout << std::endl;
    }

    // Cleanup
    if (dWorkspace) cudaFree(dWorkspace);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matA));
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matB));
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matC));
    CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan));
    CHECK_CUSPARSE(cusparseLtDestroy(&handle));

    std::cout << "SpMM using cuSPARSELt completed successfully." << std::endl;

    free(h_C);

    return 0;
}
