#pragma once

#include <string>
#include "llama_backend.hpp"

// Top-level application config
struct AppConfig {
    // MPI info
    int world_rank = 0;
    int world_size = 1;

    // Llama backend config
    LlamaBackendConfig llama_cfg;

    // General options
    bool is_gateway = false;      // rank 0
    bool verbose    = true;

    // Model topology JSON path (optional)
    std::string topology_path;
};

// Parse command-line arguments into AppConfig.
//
// Recognized flags (exemples):
//   --model PATH
//   --n_ctx N
//   --n_gpu_layers N
//   --seed N
//   --no-flash-attn
//   --quiet
//   --topology path.json
AppConfig parse_app_config(int argc, char** argv,
                           int world_rank, int world_size);

// Print a short summary of the config (for debugging/logging).
void print_app_config(const AppConfig& cfg);