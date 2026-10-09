#include "gpu/gpu_runtime.hpp"
#include "gpu_runtime_cuda_detail.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>

// -----------------------------------------------------------------------------
// Concrete implementation of the opaque common GPU stream for CUDA.
//
// GpuStream is defined publicly as:
//   struct GpuStreamImpl;
//   using GpuStream = GpuStreamImpl*;
//
// The full layout stays private to the CUDA backend.
// -----------------------------------------------------------------------------
struct GpuStreamImpl {
    cudaStream_t stream = nullptr;
};

// -----------------------------------------------------------------------------
// CUDA-only helpers exposed to CUDA backend source files through
// gpu_runtime_cuda_detail.hpp.
// -----------------------------------------------------------------------------
void check_cuda(cudaError_t result, const char* what) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(what) + ": " +
            cudaGetErrorString(result));
    }
}

cudaStream_t cuda_native_stream(GpuStream stream) {
    if (stream == nullptr) {
        throw std::invalid_argument(
            "cuda_native_stream: stream must not be nullptr");
    }

    return stream->stream;
}

// -----------------------------------------------------------------------------
// Public backend-neutral GPU runtime interface implementation.
// -----------------------------------------------------------------------------
GpuBackend gpu_backend() noexcept {
    return GpuBackend::CUDA;
}

const char* gpu_backend_name() noexcept {
    return "CUDA/NCCL";
}

int gpu_device_count() {
    int count = 0;

    check_cuda(
        cudaGetDeviceCount(&count),
        "cudaGetDeviceCount");

    return count;
}

void gpu_set_device(int device) {
    const int count = gpu_device_count();

    if (device < 0 || device >= count) {
        throw std::out_of_range(
            "gpu_set_device: device index " +
            std::to_string(device) +
            " is outside [0, " +
            std::to_string(count - 1) + "]");
    }

    check_cuda(
        cudaSetDevice(device),
        "cudaSetDevice");
}

int select_device_for_rank(int world_rank) {
    if (world_rank < 0) {
        throw std::invalid_argument(
            "select_device_for_rank: world_rank must be non-negative");
    }

    const int count = gpu_device_count();

    if (count <= 0) {
        throw std::runtime_error(
            "No CUDA GPU devices are available");
    }

    // Basic single-node policy.
    //
    // For multi-node execution, replace world_rank with a node-local MPI rank,
    // otherwise ranks running on different nodes may not map as intended.
    const int device = world_rank % count;

    gpu_set_device(device);

    return device;
}

GpuDeviceInfo gpu_get_device_info(int device) {
    const int count = gpu_device_count();

    if (device < 0 || device >= count) {
        throw std::out_of_range(
            "gpu_get_device_info: device index " +
            std::to_string(device) +
            " is outside [0, " +
            std::to_string(count - 1) + "]");
    }

    cudaDeviceProp properties {};

    check_cuda(
        cudaGetDeviceProperties(&properties, device),
        "cudaGetDeviceProperties");

    GpuDeviceInfo info;
    info.id = device;
    info.name = properties.name;
    info.compute_units = properties.multiProcessorCount;
    info.global_mem_bytes =
        static_cast<std::size_t>(properties.totalGlobalMem);

    return info;
}

void print_device_info(int device) {
    const GpuDeviceInfo info = gpu_get_device_info(device);

    std::cout << "[GPU] Backend=" << gpu_backend_name()
              << ", device=" << info.id
              << ", name=\"" << info.name << "\""
              << ", SMs=" << info.compute_units
              << ", global_mem="
              << (info.global_mem_bytes / (1024ULL * 1024ULL))
              << " MiB\n";
}

GpuStream gpu_stream_create() {
    auto* stream = new GpuStreamImpl {};

    try {
        check_cuda(
            cudaStreamCreate(&stream->stream),
            "cudaStreamCreate");
    } catch (...) {
        delete stream;
        throw;
    }

    return stream;
}

void gpu_stream_destroy(GpuStream stream) {
    if (stream == nullptr) {
        return;
    }

    try {
        if (stream->stream != nullptr) {
            check_cuda(
                cudaStreamDestroy(stream->stream),
                "cudaStreamDestroy");

            stream->stream = nullptr;
        }
    } catch (...) {
        // Ensure that the C++ wrapper itself does not leak if CUDA reports
        // destruction failure. The caller still receives the exception.
        delete stream;
        throw;
    }

    delete stream;
}

void gpu_stream_synchronize(GpuStream stream) {
    const cudaStream_t cuda_stream = cuda_native_stream(stream);

    check_cuda(
        cudaStreamSynchronize(cuda_stream),
        "cudaStreamSynchronize");
}