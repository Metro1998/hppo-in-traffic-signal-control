#ifndef SUMOCLIENT_H
#define SUMOCLIENT_H

#include <libsumo/libsumo.h>
#include "traffic_light.h"
#include "retrieve_strategy.h"
#include <vector>
#include <memory>
#include <string>
#include <algorithm>

namespace libsumo {

class SumoClient {
 private:
    std::string path_to_sumo_;
    std::string net_;
    std::string route_;
    std::string addition_;
    int yellow_time_;
    int random_seed_;

    const std::string kMaxDepartDelay = "-1";
    const std::string kWaitingTimeMemory = "1000";
    const std::string kTimeToTeleport = "-1";
    const std::string kNoWarnings = "true";

    std::vector<std::string> sumo_cmd_;
    std::vector<std::unique_ptr<TrafficLight>> traffic_lights_;

    RetrieveStrategy* observation_strategy_;
    RetrieveStrategy* reward_strategy_;



 public:
    SumoClient(
        const std::string& path_to_sumo,
        const std::string& net,
        const std::string& route,
        const std::string& addition,
        int yellow_time,
        int random_seed,
        RetrieveStrategy* observation_strategy,
        RetrieveStrategy* reward_strategy
    ) : path_to_sumo_(path_to_sumo),
        net_(net),
        route_(route),
        addition_(addition),
        yellow_time_(yellow_time),
        random_seed_(random_seed),
        observation_strategy_(observation_strategy),
        reward_strategy_(reward_strategy) {
            PrepareSimulation();
            PrepareTrafficLights();
        }

    void PrepareSimulation() {
        sumo_cmd_ = {
            path_to_sumo_,
            std::string("-n"),
            net_,
            std::string("-r"),
            route_,
            std::string("-a"),
            addition_,
            std::string("--max-depart-delay"),
            kMaxDepartDelay,
            std::string("--waiting-time-memory"),
            kWaitingTimeMemory,
            std::string("--time-to-teleport"),
            kTimeToTeleport,
            std::string("--no-warnings"),
            kNoWarnings,
            std::string("--seed"),
            std::to_string(random_seed_)
        };
        Simulation::start(sumo_cmd_);
    }

    void PrepareTrafficLights() {
        std::vector<std::string> tls_ids = TrafficLight::getIDList();
        std::for_each(tls_ids.begin(), tls_ids.end(), [this](const std::string& id) {
            traffic_lights_.emplace_back(std::make_unique<TrafficLight>(id, yellow_time_));
        });
    }

    auto RetrieveObservation() {
        // Implementation...
    }
};

} // namespace libsumo

#endif // SUMOCLIENT_H