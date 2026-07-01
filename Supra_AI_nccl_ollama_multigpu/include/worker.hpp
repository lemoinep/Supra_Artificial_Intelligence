#pragma once

#include "config.hpp"
#include "model_topology.hpp"

// Main worker loop.
// - cfg: per-rank configuration
// - topology: mapping model_id -> ModelGroup
//   (the worker will find the ModelGroup(s) it belongs to)
void worker_loop(const AppConfig& cfg,
                 const ModelTopology& topology);