#pragma once

#include <vector>

// Simple representation of a LoRA adapter (flattened weights). add tensor....

struct LoRAAdapter {
    std::vector<float> weights;   // flattened A/B matrices, etc.
};