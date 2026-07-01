#include "api_http.hpp"
#include "scheduler_threadsafe.hpp"
#include <nlohmann/json.hpp>

#include "httplib.h"

using json = nlohmann::json;

namespace api {

void setup_http_routes(httplib::Server& srv,
                       ThreadSafeScheduler& scheduler,
                       ResponseStore& resp_store,
                       int& next_request_id)
{
    srv.Post("/v1/chat/completions", [&](const httplib::Request& req,
                                         httplib::Response& res) {
        try {
            json j = json::parse(req.body);
            std::string model_id = j.value("model", "default");

            std::string prompt;
            if (j.contains("messages") && j["messages"].is_array()) {
                for (auto& m : j["messages"]) {
                    std::string role    = m.value("role", "user");
                    std::string content = m.value("content", "");
                    prompt += role + ": " + content + "\n";
                }
            } else {
                prompt = j.value("prompt", "");
            }

            int max_tokens = j.value("max_tokens", 128);
            float temperature = j.value("temperature", 0.7f);
            float top_p       = j.value("top_p", 0.9f);

            Request r;
            r.id         = next_request_id++;
            r.model_id   = model_id;
            r.prompt     = prompt;
            r.priority   = 1;
            r.est_tokens = max_tokens;

            scheduler.enqueue(r);

            json reply;
            reply["status"]     = "queued";
            reply["request_id"] = r.id;
            res.set_content(reply.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json({{"error", e.what()}}).dump(),
                            "application/json");
        }
    });

    // GET /v1/request/{id}
    srv.Get(R"(/v1/request/(\d+))", [&](const httplib::Request& req,
                                       httplib::Response& res) {
        try {
            int id = std::stoi(req.matches[1]);
            auto resp_opt = resp_store.get(id);
            if (!resp_opt) {
                res.status = 404;
                res.set_content(json({{"error", "not found"}}).dump(),
                                "application/json");
                return;
            }
            json reply;
            reply["id"]      = id;
            reply["content"] = *resp_opt;
            res.set_content(reply.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json({{"error", e.what()}}).dump(),
                            "application/json");
        }
    });

    srv.Post("/admin/lora/load", [&](const httplib::Request& req,
                                     httplib::Response& res) {
        try {
            json j = json::parse(req.body);
            std::string model_id = j.value("model", "default");
            std::string adapter  = j.value("adapter", "default");
            int priority         = j.value("priority", 10);

            std::string prompt = ":load_lora " + adapter;

            Request r;
            r.id         = next_request_id++;
            r.model_id   = model_id;
            r.prompt     = prompt;
            r.priority   = priority;
            r.est_tokens = 1;

            scheduler.enqueue(r);

            json reply;
            reply["status"]     = "queued";
            reply["request_id"] = r.id;
            res.set_content(reply.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json({{"error", e.what()}}).dump(),
                            "application/json");
        }
    });
}

} // namespace api