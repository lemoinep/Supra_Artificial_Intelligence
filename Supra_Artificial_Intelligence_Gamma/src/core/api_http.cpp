#include "api_http.hpp"

#include "scheduler_threadsafe.hpp"

#include <nlohmann/json.hpp>

#include "httplib.h"

#include <atomic>
#include <charconv>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

using json = nlohmann::json;

namespace {

constexpr std::size_t kMaxPromptBytes = 1024U * 1024U;
constexpr int kDefaultMaxTokens = 128;
constexpr int kMaximumMaxTokens = 8192;

void set_json_response(httplib::Response& response,
                       int status,
                       const json& body) {
    response.status = status;
    response.set_content(
        body.dump(),
        "application/json");
}

void set_json_error(httplib::Response& response,
                    int status,
                    const std::string& message) {
    set_json_response(
        response,
        status,
        json {
            {"error", message},
        });
}

void require_json_object(const json& value,
                         const char* field_name) {
    if (!value.is_object()) {
        throw std::invalid_argument(
            std::string(field_name) +
            " must be a JSON object");
    }
}

std::string require_non_empty_string(const json& object,
                                     const char* key,
                                     const std::string& fallback = {}) {
    if (!object.contains(key)) {
        if (!fallback.empty()) {
            return fallback;
        }

        throw std::invalid_argument(
            std::string("Missing required string field '") +
            key +
            "'");
    }

    if (!object.at(key).is_string()) {
        throw std::invalid_argument(
            std::string("Field '") +
            key +
            "' must be a string");
    }

    const std::string value =
        object.at(key).get<std::string>();

    if (value.empty()) {
        throw std::invalid_argument(
            std::string("Field '") +
            key +
            "' must not be empty");
    }

    return value;
}

int optional_int(const json& object,
                 const char* key,
                 int default_value) {
    if (!object.contains(key)) {
        return default_value;
    }

    if (!object.at(key).is_number_integer()) {
        throw std::invalid_argument(
            std::string("Field '") +
            key +
            "' must be an integer");
    }

    return object.at(key).get<int>();
}

float optional_float(const json& object,
                     const char* key,
                     float default_value) {
    if (!object.contains(key)) {
        return default_value;
    }

    if (!object.at(key).is_number()) {
        throw std::invalid_argument(
            std::string("Field '") +
            key +
            "' must be a number");
    }

    return object.at(key).get<float>();
}

std::string build_prompt_from_messages(const json& messages) {
    if (!messages.is_array()) {
        throw std::invalid_argument(
            "Field 'messages' must be an array");
    }

    if (messages.empty()) {
        throw std::invalid_argument(
            "Field 'messages' must not be empty");
    }

    std::string prompt;

    for (const auto& message : messages) {
        require_json_object(message, "Each message");

        const std::string role =
            require_non_empty_string(
                message,
                "role",
                "user");

        const std::string content =
            require_non_empty_string(
                message,
                "content");

        if (prompt.size() +
                role.size() +
                content.size() +
                3U >
            kMaxPromptBytes) {
            throw std::invalid_argument(
                "Combined chat prompt exceeds maximum size");
        }

        prompt += role;
        prompt += ": ";
        prompt += content;
        prompt += '\n';
    }

    return prompt;
}

std::string parse_prompt(const json& body) {
    if (body.contains("messages")) {
        return build_prompt_from_messages(
            body.at("messages"));
    }

    return require_non_empty_string(
        body,
        "prompt");
}

int parse_request_id(std::string_view text) {
    int id = 0;

    const auto [end, error] = std::from_chars(
        text.data(),
        text.data() + text.size(),
        id);

    if (error != std::errc {} ||
        end != text.data() + text.size() ||
        id < 0) {
        throw std::invalid_argument(
            "Request ID must be a non-negative integer");
    }

    return id;
}

void validate_sampling_params(int max_tokens,
                              float temperature,
                              float top_p) {
    if (max_tokens <= 0 || max_tokens > kMaximumMaxTokens) {
        throw std::invalid_argument(
            "'max_tokens' must be in [1, " +
            std::to_string(kMaximumMaxTokens) +
            "]");
    }

    if (temperature < 0.0f || temperature > 5.0f) {
        throw std::invalid_argument(
            "'temperature' must be in [0, 5]");
    }

    if (top_p <= 0.0f || top_p > 1.0f) {
        throw std::invalid_argument(
            "'top_p' must be in (0, 1]");
    }
}

}  // namespace

namespace api {

void setup_http_routes(httplib::Server& srv,
                       ThreadSafeScheduler& scheduler,
                       ResponseStore& resp_store,
                       std::atomic<int>& next_request_id) {
    srv.Post(
        "/v1/chat/completions",
        [&](const httplib::Request& req,
            httplib::Response& res) {
            try {
                const json body =
                    json::parse(req.body);

                require_json_object(
                    body,
                    "Request body");

                const std::string model_id =
                    require_non_empty_string(
                        body,
                        "model",
                        "default");

                const std::string prompt =
                    parse_prompt(body);

                if (prompt.size() > kMaxPromptBytes) {
                    throw std::invalid_argument(
                        "Prompt exceeds maximum size");
                }

                const int max_tokens =
                    optional_int(
                        body,
                        "max_tokens",
                        kDefaultMaxTokens);

                const float temperature =
                    optional_float(
                        body,
                        "temperature",
                        0.7f);

                const float top_p =
                    optional_float(
                        body,
                        "top_p",
                        0.9f);

                validate_sampling_params(
                    max_tokens,
                    temperature,
                    top_p);

                Request request;
                request.id = next_request_id.fetch_add(
                    1,
                    std::memory_order_relaxed);

                request.model_id = model_id;
                request.prompt = prompt;
                request.priority = 1;
                request.est_tokens = max_tokens;

                // TODO: Add max_tokens, temperature and top_p as fields in
                // Request, then serialize them to the worker protocol.
                scheduler.enqueue(request);

                set_json_response(
                    res,
                    202,
                    json {
                        {"status", "queued"},
                        {"request_id", request.id},
                        {"model", model_id},
                    });
            } catch (const json::exception& error) {
                set_json_error(
                    res,
                    400,
                    std::string("Invalid JSON request: ") +
                    error.what());
            } catch (const std::exception& error) {
                set_json_error(
                    res,
                    400,
                    error.what());
            }
        });

    srv.Get(
        R"(/v1/request/(\d+))",
        [&](const httplib::Request& req,
            httplib::Response& res) {
            try {
                if (req.matches.size() < 2U) {
                    throw std::invalid_argument(
                        "Missing request ID");
                }

                const int request_id =
                    parse_request_id(
                        req.matches[1].str());

                const auto response =
                    resp_store.get(request_id);

                if (!response.has_value()) {
                    set_json_error(
                        res,
                        404,
                        "request result not found");

                    return;
                }

                set_json_response(
                    res,
                    200,
                    json {
                        {"id", request_id},
                        {"status", "completed"},
                        {"content", *response},
                    });
            } catch (const std::exception& error) {
                set_json_error(
                    res,
                    400,
                    error.what());
            }
        });

    srv.Post(
        "/admin/lora/load",
        [&](const httplib::Request& req,
            httplib::Response& res) {
            try {
                const json body =
                    json::parse(req.body);

                require_json_object(
                    body,
                    "Request body");

                const std::string model_id =
                    require_non_empty_string(
                        body,
                        "model",
                        "default");

                const std::string adapter =
                    require_non_empty_string(
                        body,
                        "adapter",
                        "default");

                const int priority =
                    optional_int(
                        body,
                        "priority",
                        10);

                if (priority < 0) {
                    throw std::invalid_argument(
                        "'priority' must be non-negative");
                }

                Request request;
                request.id = next_request_id.fetch_add(
                    1,
                    std::memory_order_relaxed);

                request.model_id = model_id;
                request.prompt = ":load_lora " + adapter;
                request.priority = priority;
                request.est_tokens = 1;

                // This only enqueues intent. The dispatcher must recognize
                // this command and broadcast it to every rank in the selected
                // ModelGroup before any NCCL/RCCL collective is invoked.
                scheduler.enqueue(request);

                set_json_response(
                    res,
                    202,
                    json {
                        {"status", "queued"},
                        {"request_id", request.id},
                        {"model", model_id},
                        {"adapter", adapter},
                    });
            } catch (const json::exception& error) {
                set_json_error(
                    res,
                    400,
                    std::string("Invalid JSON request: ") +
                    error.what());
            } catch (const std::exception& error) {
                set_json_error(
                    res,
                    400,
                    error.what());
            }
        });
}

}  // namespace api