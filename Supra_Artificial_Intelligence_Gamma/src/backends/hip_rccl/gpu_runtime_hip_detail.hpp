#pragma once

#include "gpu/gpu_runtime.hpp"

#include <hip/hip_runtime.h>

// -----------------------------------------------------------------------------
// Private HIP runtime helpers.
//
// This header is only for implementation files in:
//
//   src/backends/hip_rccl/
//
// Do not include it from src/core/ or from public headers under include/.
// -----------------------------------------------------------------------------

// Convert the backend-neutral opaque GpuStream handle into a native HIP stream.
//
// Throws std::invalid_argument if stream is nullptr.
hipStream_t hip_native_stream(GpuStream stream);

// Check a HIP Runtime API result and throw std::runtime_error on failure.
//
// Example:
//   check_hip(hipMalloc(&buffer, bytes), "hipMalloc buffer");
void check_hip(hipError_t result, const char* what);

