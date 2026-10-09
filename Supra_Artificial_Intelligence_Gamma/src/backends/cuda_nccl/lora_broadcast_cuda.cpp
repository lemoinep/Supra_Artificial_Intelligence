#include "gpu/lora_broadcast.hpp"

#include "gpu_runtime_cuda_detail.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>

// -----------------------------------------------------------------------------
// CUDA + NCCL implementation of LoRA adapter broadcast.
//
// Preconditions:
//   - Every rank in model_comm calls this function in the same order.
//   - Every rank has adapter.weights resized to the same element count.
//   - root_local_rank is a communicator rank, not a CUDA device ID.
// -----------------------------------------------------------------------------
void broadcast_lora_adapter(LoRAAdapter& adapter,
                            ncclComm_t model_comm,
                            int root_local_rank,
                            GpuStream stream) {
    if (model_comm == nullptr) {
        throw std::invalid_argument(
            "broadcast_lora_adapter: model_comm must not be nullptr");
    }

    if (stream == nullptr) {
        throw std::invalid_argument(
            "broadcast_lora_adapter: stream must not be nullptr");
    }

    if (root_local_rank < 0) {
        throw std::invalid_argument(
            "broadcast_lora_adapter: root_local_rank must be non-negative");
    }

    const std::size_t count = adapter.element_count();

    if (count == 0) {
        throw std::runtime_error(
            "broadcast_lora_adapter: adapter has no weights");
    }

    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error(
            "broadcast_lora_adapter: LoRA buffer byte size overflows");
    }

    const std::size_t bytes = adapter.byte_size();

    float* device_weights = nullptr;
    const cudaStream_t cuda_stream = cuda_native_stream(stream);

    try {
        // Allocate one temporary device buffer per participating rank.
        check_cuda(
            cudaMalloc(&device_weights, bytes),
            "cudaMalloc LoRA device buffer");

        // Root copies valid LoRA values; non-root values are overwritten by
        // ncclBroadcast, but a same-sized host buffer exists on every rank.
        check_cuda(
            cudaMemcpyAsync(device_weights,
                            adapter.weights.data(),
                            bytes,
                            cudaMemcpyHostToDevice,
                            cuda_stream),
            "cudaMemcpyAsync LoRA host-to-device");

        // In-place NCCL broadcast: root buffer is sent to all group ranks.
        check_collective(
            ncclBroadcast(device_weights,
                          device_weights,
                          count,
                          ncclFloat,
                          root_local_rank,
                          model_comm,
                          cuda_stream),
            "ncclBroadcast LoRA adapter");

        // Make the received LoRA weights available in host-side storage.
        check_cuda(
            cudaMemcpyAsync(adapter.weights.data(),
                            device_weights,
                            bytes,
                            cudaMemcpyDeviceToHost,
                            cuda_stream),
            "cudaMemcpyAsync LoRA device-to-host");

        // Required before the caller reads adapter.weights or the device buffer
        // is released.
        check_cuda(
            cudaStreamSynchronize(cuda_stream),
            "cudaStreamSynchronize LoRA broadcast");

        check_cuda(
            cudaFree(device_weights),
            "cudaFree LoRA device buffer");

        device_weights = nullptr;
    } catch (...) {
        if (device_weights != nullptr) {
            // Preserve the original exception; cleanup failure is ignored.
            (void) cudaFree(device_weights);
        }

        throw;
    }
}