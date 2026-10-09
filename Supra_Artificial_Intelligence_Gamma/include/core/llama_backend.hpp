#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Forward declarations from the llama.cpp C API.
//
// The complete llama.h API is included only in llama_backend.cpp.
// This keeps the public C++ interface independent from llama.cpp header details.
struct llama_model;
struct llama_context;

// Application-level token identifier.
using TokenId = std::int32_t;

// -----------------------------------------------------------------------------
// Configuration used to load one GGUF model and create one llama.cpp context.
// -----------------------------------------------------------------------------
struct LlamaBackendConfig {
    // Path to the GGUF model file.
    std::string model_path;

    // Context window size.
    int n_ctx = 4096;

    // Number of transformer layers to offload to the GPU.
    //
    // Exact semantics depend on the linked llama.cpp version. Commonly:
    //   0  -> CPU-only inference
    //  >0  -> partial GPU offload
    //  -1  -> attempt full model offload
    int n_gpu_layers = 35;

    // Context RNG seed. A value of 0 may mean random depending on llama.cpp.
    int seed = 0;

    // Placeholder for a future layer/tensor split configuration.
    //
    // It is intentionally not passed into llama.cpp until the implementation
    // is adapted to the exact llama.cpp API version used by this project.
    int gpu_layer_split_mode = 0;

    // Request Flash Attention if supported by the linked llama.cpp build.
    bool use_flash_attn = true;

    // Print model information while loading.
    bool verbose = true;
};

// -----------------------------------------------------------------------------
// Per-generation sampling parameters.
// -----------------------------------------------------------------------------
struct GenerationParams {
    int max_tokens = 128;

    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;

    float repeat_penalty = 1.0f;

    // Optional per-request seed.
    //
    // The initial implementation may use the context seed only. Do not assume
    // this value has an effect until llama_backend.cpp applies it explicitly.
    int seed = 0;
};

// -----------------------------------------------------------------------------
// RAII wrapper around llama.cpp's llama_model and llama_context objects.
//
// This class remains independent from the CUDA/HIP selection. The selected
// llama.cpp library determines whether the actual execution backend is CPU,
// CUDA, or ROCm/HIP.
// -----------------------------------------------------------------------------
class LlamaBackend {
public:
    explicit LlamaBackend(const LlamaBackendConfig& config);
    ~LlamaBackend() noexcept;

    LlamaBackend(const LlamaBackend&) = delete;
    LlamaBackend& operator=(const LlamaBackend&) = delete;

    LlamaBackend(LlamaBackend&&) = delete;
    LlamaBackend& operator=(LlamaBackend&&) = delete;

    // Generate text from a prompt.
    //
    // The returned string contains only generated completion text and not the
    // original prompt.
    std::string generate(const std::string& prompt,
                         const GenerationParams& generation_params);

    // Generate token IDs from already tokenized prompt IDs.
    std::vector<TokenId> generate_tokens(
        const std::vector<TokenId>& prompt_tokens,
        const GenerationParams& generation_params);

    // Read-only access for diagnostics or future advanced integration.
    [[nodiscard]] llama_model* model() const noexcept {
        return model_;
    }

    [[nodiscard]] llama_context* context() const noexcept {
        return context_;
    }

    [[nodiscard]] const LlamaBackendConfig& config() const noexcept {
        return config_;
    }

    [[nodiscard]] bool initialized() const noexcept {
        return initialized_;
    }

private:
    void init_model_and_context();

    std::vector<TokenId> tokenize(const std::string& text,
                                  bool add_bos);

    std::string detokenize(const std::vector<TokenId>& tokens);

private:
    LlamaBackendConfig config_;

    llama_model* model_ = nullptr;
    llama_context* context_ = nullptr;

    bool initialized_ = false;
};