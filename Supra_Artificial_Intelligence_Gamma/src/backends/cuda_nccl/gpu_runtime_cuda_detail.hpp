#pragma once

#include "gpu/gpu_runtime.hpp"

#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Private CUDA runtime helpers.
//
// This header is only for implementation files in:
//
//   src/backends/cuda_nccl/
//
// Do not include it from src/core/ or from public headers under include/.
// -----------------------------------------------------------------------------

// Convert the backend-neutral opaque stream handle into a CUDA stream.
//
// Throws std::invalid_argument if stream is nullptr.
cudaStream_t cuda_native_stream(GpuStream stream);

// Check a CUDA Runtime API result and throw std::runtime_error on failure.
//
// Example:
//   check_cuda(cudaMalloc(&buffer, bytes), "cudaMalloc buffer");
void check_cuda(cudaError_t result, const char* what);