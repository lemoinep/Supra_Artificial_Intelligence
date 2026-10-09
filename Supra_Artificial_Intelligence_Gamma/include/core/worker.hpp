#pragma once

#include "core/config.hpp"
#include "core/model_topology.hpp"

// -----------------------------------------------------------------------------
// Main MPI worker loop.
//
// Called only by MPI worker ranks, normally world ranks >= 1.
//
// Parameters:
//   cfg:
//     Per-rank application configuration.
//
//   topology:
//     Mapping from logical model_id to ModelGroup. The worker retrieves the
//     model groups to which its MPI rank belongs.
//
// GPU backend selection is hidden behind the shared GPU interfaces and the
// executable selected at build time:
//   - supra_ai_cuda: CUDA + NCCL
//   - supra_ai_hip : HIP + RCCL
// -----------------------------------------------------------------------------
void worker_loop(const AppConfig& cfg,
                 const ModelTopology& topology);