#include "llama_backend.hpp"

#include "llama.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

void check_llama(int code, const char* what) {
    if (code != 0) {
        throw std::runtime_error(
            std::string(what) +
            " failed with code " +
            std::to_string(code));
    }
}

void validate_backend_config(const LlamaBackendConfig& cfg) {
    if (cfg.model_path.empty()) {
        throw std::invalid_argument(
            "LlamaBackend: model_path must not be empty");
    }

    if (cfg.n_ctx <= 0) {
        throw std::invalid_argument(
            "LlamaBackend: n_ctx must be greater than zero");
    }

    if (cfg.n_gpu_layers < -1) {
        throw std::invalid_argument(
            "LlamaBackend: n_gpu_layers must be >= -1");
    }
}

void validate_generation_params(const GenerationParams& params) {
    if (params.max_tokens < 0) {
        throw std::invalid_argument(
            "GenerationParams: max_tokens must be non-negative");
    }

    if (params.temperature < 0.0f) {
        throw std::invalid_argument(
            "GenerationParams: temperature must be non-negative");
    }

    if (params.top_k < 0) {
        throw std::invalid_argument(
            "GenerationParams: top_k must be non-negative");
    }

    if (params.top_p < 0.0f || params.top_p > 1.0f) {
        throw std::invalid_argument(
            "GenerationParams: top_p must be in [0, 1]");
    }

    if (params.repeat_penalty <= 0.0f) {
        throw std::invalid_argument(
            "GenerationParams: repeat_penalty must be greater than zero");
    }
}

std::uint32_t make_sampler_seed(int configured_seed) {
    if (configured_seed != 0) {
        return static_cast<std::uint32_t>(configured_seed);
    }

    std::random_device random_device;

    return (static_cast<std::uint32_t>(random_device()) << 1U) ^
           static_cast<std::uint32_t>(random_device());
}

class BatchGuard {
public:
    explicit BatchGuard(llama_batch batch)
        : batch_(batch) {
    }

    BatchGuard(const BatchGuard&) = delete;
    BatchGuard& operator=(const BatchGuard&) = delete;

    ~BatchGuard() {
        llama_batch_free(batch_);
    }

    [[nodiscard]] llama_batch& get() noexcept {
        return batch_;
    }

private:
    llama_batch batch_;
};

class SamplerGuard {
public:
    explicit SamplerGuard(llama_sampler* sampler)
        : sampler_(sampler) {
    }

    SamplerGuard(const SamplerGuard&) = delete;
    SamplerGuard& operator=(const SamplerGuard&) = delete;

    ~SamplerGuard() {
        if (sampler_ != nullptr) {
            llama_sampler_free(sampler_);
        }
    }

    [[nodiscard]] llama_sampler* get() const noexcept {
        return sampler_;
    }

private:
    llama_sampler* sampler_ = nullptr;
};

}  // namespace

LlamaBackend::LlamaBackend(const LlamaBackendConfig& cfg)
    : cfg_(cfg) {
    init_model_and_context();
}

LlamaBackend::~LlamaBackend() {
    if (ctx_ != nullptr) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }

    if (model_ != nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
    }

    // llama_backend_free() is intentionally not called here.
    //
    // llama.cpp backend lifetime is process-global. It should be initialized
    // once during application startup and released once during shutdown,
    // after all LlamaBackend instances have been destroyed.
}

void LlamaBackend::init_model_and_context() {
    if (initialized_) {
        return;
    }

    validate_backend_config(cfg_);

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = cfg_.n_gpu_layers;

    if (cfg_.verbose) {
        std::cout << "[LlamaBackend] Loading model from: "
                  << cfg_.model_path
                  << '\n';

        std::cout << "[LlamaBackend] n_ctx="
                  << cfg_.n_ctx
                  << ", n_gpu_layers="
                  << cfg_.n_gpu_layers
                  << '\n';
    }

    model_ = llama_model_load_from_file(
        cfg_.model_path.c_str(),
        model_params);

    if (model_ == nullptr) {
        throw std::runtime_error(
            "LlamaBackend: failed to load GGUF model from '" +
            cfg_.model_path +
            "'");
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = static_cast<std::uint32_t>(cfg_.n_ctx);

    ctx_ = llama_init_from_model(model_, context_params);

    if (ctx_ == nullptr) {
        throw std::runtime_error(
            "LlamaBackend: failed to create llama context");
    }

    if (cfg_.verbose) {
        llama_model_print_info(model_);
    }

    initialized_ = true;
}

std::vector<TokenId>
LlamaBackend::tokenize(const std::string& text, bool add_bos) {
    if (!initialized_ || model_ == nullptr) {
        throw std::runtime_error(
            "LlamaBackend::tokenize: backend is not initialized");
    }

    if (text.empty()) {
        return {};
    }

    const llama_vocab* vocab = llama_model_get_vocab(model_);

    if (vocab == nullptr) {
        throw std::runtime_error(
            "LlamaBackend::tokenize: model vocabulary is unavailable");
    }

    // The API returns a negative required size if the supplied output buffer
    // is too small. Start with a practical capacity, then retry precisely.
    std::vector<llama_token> tokens(text.size() + 8U);

    int count = llama_tokenize(
        vocab,
        text.data(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        add_bos,
        true);

    if (count < 0) {
        const std::size_t required =
            static_cast<std::size_t>(-count);

        tokens.resize(required);

        count = llama_tokenize(
            vocab,
            text.data(),
            static_cast<int32_t>(text.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            add_bos,
            true);
    }

    if (count < 0) {
        throw std::runtime_error(
            "LlamaBackend::tokenize: llama_tokenize failed");
    }

    tokens.resize(static_cast<std::size_t>(count));

    return std::vector<TokenId>(
        tokens.begin(),
        tokens.end());
}

std::string
LlamaBackend::detokenize(const std::vector<TokenId>& tokens) {
    if (!initialized_ || model_ == nullptr) {
        throw std::runtime_error(
            "LlamaBackend::detokenize: backend is not initialized");
    }

    if (tokens.empty()) {
        return {};
    }

    const llama_vocab* vocab = llama_model_get_vocab(model_);

    if (vocab == nullptr) {
        throw std::runtime_error(
            "LlamaBackend::detokenize: model vocabulary is unavailable");
    }

    std::string result;
    result.reserve(tokens.size() * 4U);

    std::vector<char> piece_buffer(256);

    for (const TokenId token : tokens) {
        int piece_size = llama_token_to_piece(
            vocab,
            static_cast<llama_token>(token),
            piece_buffer.data(),
            static_cast<int32_t>(piece_buffer.size()),
            0,
            true);

        if (piece_size < 0) {
            const std::size_t required =
                static_cast<std::size_t>(-piece_size);

            piece_buffer.resize(required);

            piece_size = llama_token_to_piece(
                vocab,
                static_cast<llama_token>(token),
                piece_buffer.data(),
                static_cast<int32_t>(piece_buffer.size()),
                0,
                true);
        }

        if (piece_size < 0) {
            throw std::runtime_error(
                "LlamaBackend::detokenize: llama_token_to_piece failed");
        }

        result.append(
            piece_buffer.data(),
            static_cast<std::size_t>(piece_size));
    }

    return result;
}

std::vector<TokenId>
LlamaBackend::generate_tokens(
    const std::vector<TokenId>& prompt_tokens,
    const GenerationParams& gen_params) {
    if (!initialized_ || ctx_ == nullptr || model_ == nullptr) {
        throw std::runtime_error(
            "LlamaBackend::generate_tokens: backend is not initialized");
    }

    validate_generation_params(gen_params);

    if (prompt_tokens.empty()) {
        throw std::invalid_argument(
            "LlamaBackend::generate_tokens: prompt must contain at least one token");
    }

    const std::size_t context_size =
        static_cast<std::size_t>(cfg_.n_ctx);

    if (prompt_tokens.size() > context_size) {
        throw std::invalid_argument(
            "LlamaBackend::generate_tokens: prompt exceeds n_ctx");
    }

    const std::size_t remaining_context =
        context_size - prompt_tokens.size();

    const std::size_t requested_tokens =
        static_cast<std::size_t>(gen_params.max_tokens);

    const std::size_t max_new_tokens =
        std::min(requested_tokens, remaining_context);

    if (max_new_tokens == 0) {
        return {};
    }

    // A context is reused by the worker, so previous request state must be
    // removed before decoding this request.
    llama_memory_clear(llama_get_memory(ctx_), true);

    std::vector<llama_token> prompt(
        prompt_tokens.begin(),
        prompt_tokens.end());

    BatchGuard prompt_batch(
        llama_batch_init(
            static_cast<int32_t>(prompt.size()),
            0,
            1));

    llama_batch& batch = prompt_batch.get();
    batch.n_tokens = static_cast<int32_t>(prompt.size());

    for (int32_t index = 0; index < batch.n_tokens; ++index) {
        batch.token[index] = prompt[static_cast<std::size_t>(index)];
        batch.pos[index] = index;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = 0;
        batch.logits[index] = false;
    }

    // Only logits of the final prompt token are needed to sample token 1.
    batch.logits[batch.n_tokens - 1] = true;

    check_llama(
        llama_decode(ctx_, batch),
        "llama_decode(prompt)");

    llama_sampler_chain_params sampler_params =
        llama_sampler_chain_default_params();

    SamplerGuard sampler(
        llama_sampler_chain_init(sampler_params));

    if (sampler.get() == nullptr) {
        throw std::runtime_error(
            "LlamaBackend::generate_tokens: failed to create sampler chain");
    }

    if (gen_params.repeat_penalty != 1.0f) {
        llama_sampler_chain_add(
            sampler.get(),
            llama_sampler_init_penalties(
                64,
                gen_params.repeat_penalty,
                0.0f,
                0.0f));
    }

    if (gen_params.top_k > 0) {
        llama_sampler_chain_add(
            sampler.get(),
            llama_sampler_init_top_k(gen_params.top_k));
    }

    if (gen_params.top_p > 0.0f && gen_params.top_p < 1.0f) {
        llama_sampler_chain_add(
            sampler.get(),
            llama_sampler_init_top_p(
                gen_params.top_p,
                1));
    }

    if (gen_params.temperature > 0.0f) {
        llama_sampler_chain_add(
            sampler.get(),
            llama_sampler_init_temp(gen_params.temperature));
    } else {
        llama_sampler_chain_add(
            sampler.get(),
            llama_sampler_init_greedy());
    }

    if (gen_params.temperature > 0.0f) {
        llama_sampler_chain_add(
            sampler.get(),
            llama_sampler_init_dist(
                make_sampler_seed(cfg_.seed)));
    }

    std::vector<TokenId> output_tokens;
    output_tokens.reserve(max_new_tokens);

    llama_token next_token = llama_sampler_sample(
        sampler.get(),
        ctx_,
        -1);

    for (std::size_t generated = 0;
         generated < max_new_tokens;
         ++generated) {
        if (llama_vocab_is_eog(
                llama_model_get_vocab(model_),
                next_token)) {
            break;
        }

        output_tokens.push_back(
            static_cast<TokenId>(next_token));

        llama_sampler_accept(
            sampler.get(),
            next_token);

        if (generated + 1U >= max_new_tokens) {
            break;
        }

        llama_token token_for_decode = next_token;

        llama_batch next_batch = llama_batch_get_one(
            &token_for_decode,
            1);

        check_llama(
            llama_decode(ctx_, next_batch),
            "llama_decode(next token)");

        next_token = llama_sampler_sample(
            sampler.get(),
            ctx_,
            -1);
    }

    return output_tokens;
}

std::string
LlamaBackend::generate(const std::string& prompt,
                       const GenerationParams& gen_params) {
    const std::vector<TokenId> prompt_tokens =
        tokenize(prompt, true);

    const std::vector<TokenId> generated_tokens =
        generate_tokens(prompt_tokens, gen_params);

    return detokenize(generated_tokens);
}