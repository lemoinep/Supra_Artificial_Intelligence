#pragma once

#if defined(USE_HIP)
#include <hip/hip_runtime.h>
using GpuError = hipError_t;
inline void checkCuda(GpuError err, const char* msg) {
    if (err != hipSuccess) {
        throw std::runtime_error(
            std::string(msg) + ": " + hipGetErrorString(err));
    }
}
#else
#include <cuda_runtime.h>
using GpuError = cudaError_t;
inline void checkCuda(GpuError err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(msg) + ": " + cudaGetErrorString(err));
    }
}
#endif

int select_device_for_rank(int world_rank);
void print_device_info(int device);