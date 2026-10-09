#pragma once

#include "gpu/gpu_runtime.hpp"

#include <cuda_runtime.h>

cudaStream_t cuda_native_stream(GpuStream stream);

void check_cuda(cudaError_t result, const char* what);