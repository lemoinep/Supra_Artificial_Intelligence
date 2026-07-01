#include "llama_backend.hpp"
#include "llama.h"  

#include <stdexcept>
#include <sstream>
#include <iostream>
#include <random>
#include <algorithm>

static inline void check_llama(int code, const char* what) {
    if (code != 0) {
        std::ostringstream oss;
        oss << what << " failed with code " << code;
        throw std::runtime_error(oss.str());
    }
}

LlamaBackend::LlamaBackend(const LlamaBackendConfig& cfg)
    : cfg_(cfg)
{
    init_model_and_context();
}

LlamaBackend::~LlamaBackend() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_free_model(model_);
        model_ = nullptr;
    }
    llama_backend_free();
}

void LlamaBackend::init_model_and_context() {
    if (initialized_) return;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers    = cfg_.n_gpu_layers;
    mparams.use_flash_attn  = cfg_.use_flash_attn;
    // mparams.split_mode   = cfg_.gpu_layer_split_mode; // if dispo..

    if (cfg_.verbose) {
        std::cout << "[LlamaBackend] Loading model from: "
                  << cfg_.model_path << std::endl;
        std::cout << "[LlamaBackend] n_ctx=" << cfg_.n_ctx
                  << ", n_gpu_layers=" << cfg_.n_gpu_layers << std::endl;
    }

    model_ = llama_load_model_from_file(cfg_.model_path.c_str(), mparams);
    if (!model_) {
        throw std::runtime_error("Failed to load llama model");
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx          = cfg_.n_ctx;
    cparams.seed           = cfg_.seed;
    cparams.use_flash_attn = cfg_.use_flash_attn;

    ctx_ = llama_new_context_with_model(model_, cparams);
    if (!ctx_) {
        throw std::runtime_error("Failed to create llama context");
    }

    if (cfg_.verbose) {
        llama_model_print_info(model_);
    }

    initialized_ = true;
}

std::vector<TokenId> LlamaBackend::tokenize(const std::string& text, bool add_bos) {
    std::vector<llama_token> tmp(text.size() + 128);
    int n = llama_tokenize(
        model_,
        text.c_str(),
        tmp.data(),
        (int)tmp.size(),
        add_bos,
        true  // allow special tokens
    );
    if (n < 0) {
        throw std::runtime_error("llama_tokenize failed");
    }
    tmp.resize(n);
    std::vector<TokenId> out(tmp.begin(), tmp.end());
    return out;
}

std::string LlamaBackend::detokenize(const std::vector<TokenId>& tokens) {
    std::string result;
    result.reserve(tokens.size() * 4);
    for (auto t : tokens) {
        const char* s = llama_token_to_str(model_, t);
        if (s != nullptr) {
            result += s;
        }
    }
    return result;
}

std::vector<TokenId> LlamaBackend::generate_tokens(
    const std::vector<TokenId>& prompt_tokens,
    const GenerationParams& gen_params)
{
    if (!initialized_) {
        throw std::runtime_error("LlamaBackend not initialized");
    }

    int n_past   = 0;
    int n_prompt = (int)prompt_tokens.size();

    check_llama(
        llama_eval(ctx_,
                   prompt_tokens.data(),
                   n_prompt,
                   n_past,
                   cfg_.n_ctx),
        "llama_eval(prompt)");

    n_past += n_prompt;

    std::vector<TokenId> out_tokens;
    out_tokens.reserve(gen_params.max_tokens);

    for (int i = 0; i < gen_params.max_tokens; ++i) {
        const float* logits = llama_get_logits(ctx_);
        int vocab_size = llama_n_vocab(model_);

        std::vector<llama_token_data> candidates;
        candidates.reserve(vocab_size);
        for (int token_id = 0; token_id < vocab_size; ++token_id) {
            candidates.push_back(
                { token_id, logits[token_id], 0.0f }
            );
        }

        llama_token_data_array candidates_array = {
            candidates.data(),
            (size_t)candidates.size(),
            false
        };

        llama_sample_temperature(ctx_, &candidates_array, gen_params.temperature);
        llama_sample_top_k(ctx_, &candidates_array, gen_params.top_k, 1);
        llama_sample_top_p(ctx_, &candidates_array, gen_params.top_p, 1);

        if (!out_tokens.empty()) {
            llama_sample_repetition_penalty(
                ctx_,
                &candidates_array,
                out_tokens.data(),
                out_tokens.size(),
                gen_params.repeat_penalty
            );
        }

        llama_token new_token = llama_sample_token(ctx_, &candidates_array);
        if (new_token == llama_token_eos(model_)) {
            break;
        }

        out_tokens.push_back(new_token);

        check_llama(
            llama_eval(ctx_,
                       &new_token,
                       1,
                       n_past,
                       cfg_.n_ctx),
            "llama_eval(next_token)");

        ++n_past;
    }

    return out_tokens;
}

std::string LlamaBackend::generate(const std::string& prompt,
                                   const GenerationParams& gen_params)
{
    auto prompt_tokens = tokenize(prompt, /*add_bos=*/true);
    auto gen_tokens    = generate_tokens(prompt_tokens, gen_params);
    return detokenize(gen_tokens);
}