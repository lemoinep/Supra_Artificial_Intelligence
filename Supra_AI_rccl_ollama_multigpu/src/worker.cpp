#include "worker.hpp"
#include "gpu_utils.hpp"
#include "llama_backend.hpp"
#include "model_topology.hpp"
#include "lora_nccl.hpp"

#include <mpi.h>
#include <map>
#include <memory>
#include <iostream>

void worker_loop(const AppConfig& cfg,
    const ModelTopology& topology)
{
    int dev = select_device_for_rank(cfg.world_rank);
    if (cfg.verbose) {
        print_device_info(dev);
        std::cout << "[Worker] Rank " << cfg.world_rank
            << " using device " << dev << std::endl;
    }

    const ModelGroup* default_group = topology.get_group("default");

    ncclComm_t model_comm = nullptr;
    int mpi_rank_local = -1;
    if (default_group && default_group->mpi_comm != MPI_COMM_NULL) {
        model_comm = default_group->nccl_comm;
        mpi_rank_local = default_group->mpi_rank_local;
    }

    std::map<std::string, std::unique_ptr<LlamaBackend>> backends;

    auto get_backend_for_model = [&](const std::string& model_id) -> LlamaBackend& {
        auto it = backends.find(model_id);
        if (it != backends.end()) {
            return *(it->second);
        }
        LlamaBackendConfig lc = cfg.llama_cfg;
        if (model_id == "code") {
            lc.model_path = "code_model.gguf";
        }
        else if (model_id == "chat") {
            lc.model_path = "chat_model.gguf";
        }
        auto ptr = std::make_unique<LlamaBackend>(lc);
        auto& ref = *ptr;
        backends[model_id] = std::move(ptr);
        return ref;
        };

    LoRAAdapter lora_adapter;

#if defined(USE_HIP)
    hipStream_t lora_stream;
    checkCuda(hipStreamCreate(&lora_stream), "hipStreamCreate lora");
#else
    cudaStream_t lora_stream;
    checkCuda(cudaStreamCreate(&lora_stream), "cudaStreamCreate lora");
#endif

    GenerationParams gen_params;
    gen_params.max_tokens = 128;
    gen_params.temperature = 0.7f;
    gen_params.top_p = 0.9f;
    gen_params.top_k = 40;
    gen_params.repeat_penalty = 1.0f;

    while (true) {
        int req_id = 0;
        int model_len = 0;
        int prompt_len = 0;
        MPI_Status st;

        MPI_Recv(&req_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &st);
        if (req_id < 0) {
            if (cfg.verbose) {
                std::cout << "[Worker] Rank " << cfg.world_rank
                    << " received exit signal." << std::endl;
            }
            break;
        }

        MPI_Recv(&model_len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &st);
        std::string model_id(model_len, '\0');
        MPI_Recv(model_id.data(), model_len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &st);

        MPI_Recv(&prompt_len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &st);
        std::string prompt(prompt_len, '\0');
        MPI_Recv(prompt.data(), prompt_len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &st);

        if (cfg.verbose) {
            std::cout << "[Worker] Rank " << cfg.world_rank
                << " processing req " << req_id
                << " model_id=" << model_id
                << " prompt=\"" << prompt << "\""
                << std::endl;
        }

        std::string out;

        if (prompt.rfind(":load_lora", 0) == 0) {
            if (cfg.verbose) {
                std::cout << "[Worker] Rank " << cfg.world_rank
                    << " handling LoRA load command." << std::endl;
            }
            if (model_comm != nullptr) {
                if (mpi_rank_local == 0) {
                    lora_adapter.weights.assign(1024, 1.0f); 
                }

                broadcast_lora_adapter(lora_adapter,
                    model_comm,
                    /*root_local_rank*/ 0,
                    lora_stream);
            }
            out = "[LoRA] Adapter synchronized across model group.";
        }
        else {
            LlamaBackend& backend = get_backend_for_model(model_id);
            try {
                out = backend.generate(prompt, gen_params);
            }
            catch (const std::exception& e) {
                std::cerr << "[Worker] Rank " << cfg.world_rank
                    << " generation error: " << e.what() << std::endl;
                out = std::string("[error] ") + e.what();
            }
        }

        int out_len = (int)out.size();
        MPI_Send(&req_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&out_len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        if (out_len > 0) {
            MPI_Send(out.data(), out_len, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }

#if defined(USE_HIP)
    hipStreamDestroy(lora_stream);
#else
    cudaStreamDestroy(lora_stream);
#endif
}