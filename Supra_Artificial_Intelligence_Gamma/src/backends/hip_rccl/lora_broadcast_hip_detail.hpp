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
            "broadcast_lora_adapter: LoRA buffer byte size overflows");
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
// Preconditions:
//   - Each rank in model_comm calls this collective in the same order.
//   - Each rank has adapter.weights resized to the same number of floats.
//   - root_local_rank is a rank in model_comm, not a HIP device index.
// -----------------------------------------------------------------------------
void broadcast_lora_adapter(LoRAAdapter& adapter,
                            ncclComm_t model_comm,
                            int root_local_rank,
                            GpuStream stream) {
    const std::size_t count = adapter.element_count();

    validate_lora_broadcast_arguments(model_comm,
                                      root_local_rank,
                                      stream,
                                      count);

    const std::size_t bytes = adapter.byte_size();
    const hipStream_t hip_stream = hip_native_stream(stream);

    float* device_weights = nullptr;

    try {
        check_hip(
            hipMalloc(&device_weights, bytes),
            "hipMalloc LoRA device buffer");

        // The root buffer contains source values. Non-root input contents are
        // overwritten by the in-place RCCL broadcast, but every rank supplies
        // an equally sized device receive buffer.
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

        // Guarantees that the collective and D2H transfer are complete before
        // the caller observes adapter.weights or the buffer is freed.
        check_hip(
            hipStreamSynchronize(hip_stream),
            "hipStreamSynchronize LoRA broadcast");

        check_hip(
            hipFree(device_weights),
            "hipFree LoRA device buffer");

        device_weights = nullptr;
    } catch (...) {
        if (device_weights != nullptr) {
            // Preserve the primary error; this is cleanup only.
            (void) hipFree(device_weights);
        }

        throw;
    }
}