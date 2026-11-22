#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

static void cuda_ok(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}

static void blas_ok(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS failure %d\n", static_cast<int>(status));
        std::exit(1);
    }
}

int main() {
    constexpr size_t size = 512ul * 1024 * 1024;
    constexpr int copies = 40;
    constexpr int n = 4096;
    constexpr int products = 20;
    void *a = nullptr, *b = nullptr, *c = nullptr;
    cudaEvent_t start, end;
    cuda_ok(cudaSetDevice(0));
    cuda_ok(cudaMalloc(&a, size));
    cuda_ok(cudaMalloc(&b, size));
    cuda_ok(cudaMalloc(&c, size_t(n) * n * sizeof(__half)));
    cuda_ok(cudaMemset(a, 1, size));
    cuda_ok(cudaMemset(b, 2, size));
    cuda_ok(cudaEventCreate(&start));
    cuda_ok(cudaEventCreate(&end));
    cuda_ok(cudaMemcpyAsync(b, a, size, cudaMemcpyDeviceToDevice));
    cuda_ok(cudaEventRecord(start));
    for (int step = 0; step < copies; ++step)
        cuda_ok(cudaMemcpyAsync(step % 2 ? a : b, step % 2 ? b : a, size, cudaMemcpyDeviceToDevice));
    cuda_ok(cudaEventRecord(end));
    cuda_ok(cudaEventSynchronize(end));
    float copy_ms = 0;
    cuda_ok(cudaEventElapsedTime(&copy_ms, start, end));

    cublasHandle_t handle;
    blas_ok(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;
    auto gemm = [&]() {
        blas_ok(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                            a, CUDA_R_16F, n, b, CUDA_R_16F, n, &beta,
                            c, CUDA_R_16F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    gemm();
    cuda_ok(cudaEventRecord(start));
    for (int step = 0; step < products; ++step) gemm();
    cuda_ok(cudaEventRecord(end));
    cuda_ok(cudaEventSynchronize(end));
    float gemm_ms = 0;
    cuda_ok(cudaEventElapsedTime(&gemm_ms, start, end));
    std::printf("{\"cuda_copy_read_plus_write_GB_s\":%.3f,"
                "\"fp16_gemm_TFLOP_s\":%.3f,\"gemm_dimension\":%d}\n",
                2.0 * copies * size / copy_ms / 1e6,
                2.0 * products * n * n * n / gemm_ms / 1e9, n);
    blas_ok(cublasDestroy(handle));
    cuda_ok(cudaEventDestroy(end));
    cuda_ok(cudaEventDestroy(start));
    cuda_ok(cudaFree(c));
    cuda_ok(cudaFree(b));
    cuda_ok(cudaFree(a));
}
