#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// Check helper for CUDA calls
inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// Select a GPU for a given MPI rank.
// Simple default: rank % device_count.
int select_device_for_rank(int world_rank);

// Optionally, print device info (name, SMs, memory).
void print_device_info(int device);

