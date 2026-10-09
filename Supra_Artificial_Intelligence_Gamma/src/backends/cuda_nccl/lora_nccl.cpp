#include "lora_nccl.hpp"
#include "nccl_utils.hpp"

void broadcast_lora_adapter(LoRAAdapter& adapter,
                            ncclComm_t  model_comm,
                            int         root_local_rank,
                            cudaStream_t stream)
{
    if (model_comm == nullptr) return;

    size_t count = adapter.weights.size();
    if (count == 0) return;

    float* d_buf = nullptr;
    checkCuda(cudaMalloc(&d_buf, count * sizeof(float)), "cudaMalloc lora adapter");

    checkCuda(
        cudaMemcpyAsync(
            d_buf,
            adapter.weights.data(),
            count * sizeof(float),
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpyAsync lora host->dev");
    checkCuda(cudaStreamSynchronize(stream), "sync before ncclBcast");

    checkNccl(
        ncclBroadcast(
            d_buf, d_buf, count, ncclFloat,
            root_local_rank, model_comm, stream),
        "ncclBroadcast lora adapter");
    checkCuda(cudaStreamSynchronize(stream), "sync after ncclBcast");

    checkCuda(
        cudaMemcpyAsync(
            adapter.weights.data(),
            d_buf,
            count * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream),
        "cudaMemcpyAsync lora dev->host");
    checkCuda(cudaStreamSynchronize(stream), "sync after dev->host");

    cudaFree(d_buf);
}