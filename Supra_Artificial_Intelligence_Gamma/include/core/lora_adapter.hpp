#pragma once

#include <cstddef>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// Host-side representation of a LoRA adapter.
//
// Initial version:
// - Stores flattened FP32 weights.
// - The root rank loads/populates `weights`.
// - All other ranks resize `weights` to the same element count before
//   broadcast_lora_adapter() is called.
//
// A later version can add named tensors, shapes, rank/alpha metadata,
// a source path, a version, and a checksum.
// -----------------------------------------------------------------------------
struct LoRAAdapter {
    std::string name;
    std::vector<float> weights;

    [[nodiscard]] bool empty() const noexcept {
        return weights.empty();
    }

    [[nodiscard]] std::size_t element_count() const noexcept {
        return weights.size();
    }

    [[nodiscard]] std::size_t byte_size() const noexcept {
        return weights.size() * sizeof(float);
    }

    void clear() {
        name.clear();
        weights.clear();
    }
};