#pragma once

#include <cstddef>
#include <string>

// -----------------------------------------------------------------------------
// Common GPU runtime interface.
//
// The implementation is selected at build time:
//
//   CUDA/NCCL target:
//     src/backends/cuda_nccl/gpu_runtime_cuda.cpp
//
//   HIP/RCCL target:
//     src/backends/hip_rccl/gpu_runtime_hip.cpp
//
// This public interface must not include cuda_runtime.h or hip_runtime.h.
// -----------------------------------------------------------------------------

enum class GpuBackend {
    CUDA,
    HIP
};

// Backend-neutral subset of device information.
struct GpuDeviceInfo {
    int id = -1;
    std::string name;
    int compute_units = 0;          // CUDA SMs or AMD compute units.
    std::size_t global_mem_bytes = 0;
};

// Opaque cross-backend GPU stream.
//
// The concrete representation is private to the selected backend:
//   CUDA: wraps cudaStream_t
//   HIP : wraps hipStream_t
struct GpuStreamImpl;
using GpuStream = GpuStreamImpl*;

// Return the backend compiled into the current executable.
[[nodiscard]] GpuBackend gpu_backend() noexcept;

// Human-readable backend label, for example "CUDA/NCCL" or "HIP/RCCL".
[[nodiscard]] const char* gpu_backend_name() noexcept;

// Return the number of GPU devices visible to this MPI process.
[[nodiscard]] int gpu_device_count();

// Select the current GPU device for this host thread/process.
void gpu_set_device(int device);

// Simple single-node mapping from MPI rank to visible GPU index.
//
// For multi-node production use, evolve this to use an MPI local rank rather
// than a global MPI_COMM_WORLD rank.
[[nodiscard]] int select_device_for_rank(int world_rank);

// Return basic portable device information.
[[nodiscard]] GpuDeviceInfo gpu_get_device_info(int device);

// Print a short device information line.
void print_device_info(int device);

// Create, destroy and synchronize an opaque non-default GPU stream.
[[nodiscard]] GpuStream gpu_stream_create();

void gpu_stream_destroy(GpuStream stream);

void gpu_stream_synchronize(GpuStream stream);