#include "gpu/gpu_runtime.hpp"
#include "gpu_runtime_hip_detail.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

// -----------------------------------------------------------------------------
// Private representation of the opaque GPU stream.
//
// The core only sees GpuStream, i.e. GpuStreamImpl*.
// This HIP backend owns the native hipStream_t handle.
// -----------------------------------------------------------------------------
struct GpuStreamImpl {
    hipStream_t stream = nullptr;
};

void check_hip(hipError_t result, const char* what) {
    if (result != hipSuccess) {
        throw std::runtime_error(
            std::string(what) + ": " + hipGetErrorString(result));
    }
}

hipStream_t hip_native_stream(GpuStream stream) {
    if (stream == nullptr) {
        throw std::invalid_argument(
            "hip_native_stream: stream must not be nullptr");
    }

    return stream->stream;
}

GpuBackend gpu_backend() noexcept {
    return GpuBackend::HIP;
}

const char* gpu_backend_name() noexcept {
    return "HIP/RCCL";
}

int gpu_device_count() {
    int count = 0;

    check_hip(
        hipGetDeviceCount(&count),
        "hipGetDeviceCount");

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

    check_hip(
        hipSetDevice(device),
        "hipSetDevice");
}

int select_device_for_rank(int world_rank) {
    if (world_rank < 0) {
        throw std::invalid_argument(
            "select_device_for_rank: world_rank must be non-negative");
    }

    const int count = gpu_device_count();

    if (count <= 0) {
        throw std::runtime_error(
            "No HIP/RCCL-compatible GPU device is available");
    }

    const int device = world_rank % count;

    // hipSetDevice establishes the default device for this host thread.
    // Subsequent allocations and streams created by this thread use it.
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

    hipDeviceProp_t properties {};

    check_hip(
        hipGetDeviceProperties(&properties, device),
        "hipGetDeviceProperties");

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
              << ", CUs=" << info.compute_units
              << ", global_mem="
              << (info.global_mem_bytes / (1024ULL * 1024ULL))
              << " MiB\n";
}

GpuStream gpu_stream_create() {
    auto* stream = new GpuStreamImpl {};

    try {
        check_hip(
            hipStreamCreate(&stream->stream),
            "hipStreamCreate");
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
            check_hip(
                hipStreamDestroy(stream->stream),
                "hipStreamDestroy");

            stream->stream = nullptr;
        }
    } catch (...) {
        // Preserve the HIP error while still releasing the C++ wrapper.
        delete stream;
        throw;
    }

    delete stream;
}

void gpu_stream_synchronize(GpuStream stream) {
    const hipStream_t native_stream = hip_native_stream(stream);

    check_hip(
        hipStreamSynchronize(native_stream),
        "hipStreamSynchronize");
}