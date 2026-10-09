#pragma once

#include "core/llama_backend.hpp"

#include <string>

// -----------------------------------------------------------------------------
// Top-level application configuration.
//
// This object is identical for CUDA/NCCL and HIP/RCCL builds.
// GPU backend selection is a build-time responsibility handled by CMake.
// -----------------------------------------------------------------------------
struct AppConfig {
    // MPI world information.
    int world_rank = 0;
    int world_size = 1;

    // llama.cpp model and generation backend configuration.
    LlamaBackendConfig llama_cfg;

    // Rank 0 acts as the HTTP gateway / scheduler dispatcher.
    bool is_gateway = false;

    // Enables application logging for this rank.
    bool verbose = true;

    // Optional path to a JSON model topology description.
    std::string topology_path;
};

// Parse command-line arguments into an application configuration.
//
// Recognized options:
//   --model PATH
//   --n_ctx N
//   --n_gpu_layers N
//   --ngl N
//   --seed N
//   --no-flash-attn
//   --quiet
//   --topology PATH
//   --help, -h
AppConfig parse_app_config(int argc,
                           char** argv,
                           int world_rank,
                           int world_size);

// Print a short configuration summary for debugging and logs.
// The implementation normally prints only from world rank 0.
void print_app_config(const AppConfig& cfg);