#pragma once

#include <string>
#include <vector>
#include <cstdint>

// Forward declarations from llama.cpp C API
struct llama_model;
struct llama_context;

// Token type alias
using TokenId = int32_t;

// Configuration for the Llama backend
struct LlamaBackendConfig {
    std::string model_path;   // path to GGUF model
    int         n_ctx        = 4096;   // context window
    int         n_gpu_layers = 35;     // number of layers offloaded to GPU
    int         seed         = 0;      // RNG seed (0 = random)
    int         gpu_layer_split_mode = 0; // depends on llama.cpp version
    bool        use_flash_attn = true;    // if supported by build
    bool        verbose       = true;     // log model info at load
};

// Simple generation parameters
struct GenerationParams {
    int   max_tokens    = 128;
    float temperature   = 0.7f;
    float top_p         = 0.9f;
    int   top_k         = 40;
    float repeat_penalty = 1.0f;
    int   seed          = 0;    // optional per-request seed
};

// Main backend class wrapping llama.cpp
class LlamaBackend {
public:
    explicit LlamaBackend(const LlamaBackendConfig& cfg);
    ~LlamaBackend();

    // Non-copyable
    LlamaBackend(const LlamaBackend&) = delete;
    LlamaBackend& operator=(const LlamaBackend&) = delete;

    // Generate text given a prompt.
    // Returns the generated text (not including the original prompt).
    std::string generate(const std::string& prompt,
                         const GenerationParams& gen_params);

    // Optional: expose token-level API if you want more control later
    std::vector<TokenId> generate_tokens(const std::vector<TokenId>& prompt_tokens,
                                         const GenerationParams& gen_params);

    // Access to underlying pointers (for advanced use)
    llama_model*   model() const   { return model_; }
    llama_context* context() const { return ctx_;   }

private:
    void init_model_and_context();
    std::vector<TokenId> tokenize(const std::string& text, bool add_bos);
    std::string detokenize(const std::vector<TokenId>& tokens);

private:
    LlamaBackendConfig cfg_;
    llama_model*   model_ = nullptr;
    llama_context* ctx_   = nullptr;
    bool           initialized_ = false;
};