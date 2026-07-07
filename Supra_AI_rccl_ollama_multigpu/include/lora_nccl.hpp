#pragma once

#include "lora_adapter.hpp"
#include <nccl.h>
#include <cuda_runtime.h>

// Broadcast LoRA adapter weights to all ranks in a model NCCL communicator.
// - adapter: in/out, host-side vector of weights
// - model_comm: NCCL communicator for the model group
// - root_local_rank: rank within model group that is the source (often 0)
// - stream: CUDA stream used for copies and NCCL call
void broadcast_lora_adapter(LoRAAdapter& adapter,
                            ncclComm_t  model_comm,
                            int         root_local_rank,
                            cudaStream_t stream);