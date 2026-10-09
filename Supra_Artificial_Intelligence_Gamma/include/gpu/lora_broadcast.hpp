#pragma once

#include "core/lora_adapter.hpp"
#include "gpu/gpu_collectives.hpp"
#include "gpu/gpu_runtime.hpp"

// -----------------------------------------------------------------------------
// LoRA adapter broadcast interface.
//
// Root semantics:
//   - `root_local_rank` owns valid host-side adapter.weights before the call.
//   - every other rank must have adapter.weights resized to exactly the same
//     number of float elements before the call.
//   - after success, every rank owns an identical host-side weights vector.
//
// Build-selected implementation:
//   - CUDA + NCCL: src/backends/cuda_nccl/lora_broadcast_cuda.cpp
//   - HIP  + RCCL: src/backends/hip_rccl/lora_broadcast_hip.cpp
//
// Collective rule:
// Every participant in model_comm must call this function in the same order
// with the same root_local_rank and the same weights element count.
// -----------------------------------------------------------------------------

// Broadcast host-side LoRA weights through a temporary device buffer.
//
// Parameters:
//   adapter:
//     Root rank provides loaded weights; non-root ranks provide a resized
//     destination vector.
//
//   model_comm:
//     ModelGroup GPU communicator. It is backed by NCCL or RCCL.
//
//   root_local_rank:
//     Root rank inside model_comm, generally 0. This is not a physical GPU ID.
//
//   stream:
//     Opaque stream created with gpu_stream_create().
//
// Throws std::runtime_error on NCCL/RCCL or backend runtime failures.
void broadcast_lora_adapter(LoRAAdapter& adapter,
                            ncclComm_t   model_comm,
                            int          root_local_rank,
                            GpuStream    stream);