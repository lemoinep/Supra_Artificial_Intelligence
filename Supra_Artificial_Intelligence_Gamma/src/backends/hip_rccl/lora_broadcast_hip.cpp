#include "gpu/lora_broadcast.hpp"

#include "gpu_runtime_hip_detail.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace {

void validate_lora_broadcast_count(std::size_t count) {
    if (count == 0) {
        throw std::runtime_error(
            "broadcast_lora_adapter: adapter has no weights");
    }

    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error(
            "broadcast_lora_adapter: LoRA weight buffer size overflows");
    }
}

void validate_lora_broadcast_arguments(ncclComm_t model_comm,
                                       int root_local_rank,
                                       GpuStream stream,
                                       std::size_t count) {
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

    validate_lora_broadcast_count(count);
}

}  // namespace

// -----------------------------------------------------------------------------
// HIP + RCCL LoRA adapter broadcast.
//
// Every rank in model_comm must call this function with:
//   - the same `count`;
//   - the same `root_local_rank`;
//   - the same collective ordering.
//
// root_local_rank is a communicator rank, never a HIP device ID.
// -----------------------------------------------------------------------------
void broadcast_lora_adapter(LoRAAdapter& adapter,
                            ncclComm_t model_comm,
                            int root_local_rank,
                            GpuStream stream) {
    const std::size_t count = adapter.weights.size();

    validate_lora_broadcast_arguments(model_comm,
                                      root_local_rank,
                                      stream,
                                      count);

    const std::size_t bytes = count * sizeof(float);
    const hipStream_t hip_stream = hip_native_stream(stream);

    float* device_weights = nullptr;

    try {
        check_hip(
            hipMalloc(&device_weights, bytes),
            "hipMalloc LoRA device buffer");

        // Root copies valid data. On non-root ranks, the input data is
        // overwritten by the in-place RCCL broadcast.
        check_hip(
            hipMemcpyAsync(device_weights,
                           adapter.weights.data(),
                           bytes,
                           hipMemcpyHostToDevice,
                           hip_stream),
            "hipMemcpyAsync LoRA host-to-device");

        check_collective(
            ncclBroadcast(device_weights,
                          device_weights,
                          count,
                          ncclFloat,
                          root_local_rank,
                          model_comm,
                          hip_stream),
            "rccl ncclBroadcast LoRA adapter");

        check_hip(
            hipMemcpyAsync(adapter.weights.data(),
                           device_weights,
                           bytes,
                           hipMemcpyDeviceToHost,
                           hip_stream),
            "hipMemcpyAsync LoRA device-to-host");

        // Ensures that adapter.weights is ready before this function returns.
        check_hip(
            hipStreamSynchronize(hip_stream),
            "hipStreamSynchronize LoRA broadcast");

        check_hip(
            hipFree(device_weights),
            "hipFree LoRA device buffer");

        device_weights = nullptr;
    } catch (...) {
        if (device_weights != nullptr) {
            // Do not replace the error that triggered cleanup.
            (void) hipFree(device_weights);
        }

        throw;
    }
}