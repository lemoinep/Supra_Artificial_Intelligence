#include "gpu_utils.hpp"
#include <iostream>

int select_device_for_rank(int world_rank) {
    int num_devices = 0;
    checkCuda(cudaGetDeviceCount(&num_devices), "cudaGetDeviceCount");
    if (num_devices == 0) {
        throw std::runtime_error("No CUDA devices available");
    }

    int dev = world_rank % num_devices;
    checkCuda(cudaSetDevice(dev), "cudaSetDevice");

    return dev;
}

void print_device_info(int device) {
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    std::cout << "[GPU] Using device " << device
              << " (" << prop.name << "), SMs=" << prop.multiProcessorCount
              << ", global mem=" << (prop.totalGlobalMem / (1024 * 1024)) << " MB"
              << std::endl;
}