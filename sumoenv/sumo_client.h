#ifndef SUMOCLIENT_H
#define SUMOCLIENT_H

#include <libsumo/libsumo.h>
#include "traffic_light.h"
#include "retrieve_strategy.h"
#include <vector>
#include <memory>
#include <string>
#include <algorithm>

using Simulation = libsumo::Simulation;
using string = std::string;
template <typename T>
using vector = std::vector<T>;

class SumoClient {
 private:
    string path_to_sumo_;
    string net_;
    string route_;
    string addition_;
    int yellow_time_;
    int random_seed_;

    const string kMaxDepartDelay = "-1";
    const string kWaitingTimeMemory = "1000";
    const string kTimeToTeleport = "-1";
    const string kNoWarnings = "true";

    vector<string> sumo_cmd_;
    vector<std::unique_ptr<TrafficLight>> traffic_lights_;

    RetrieveStrategy* observation_strategy_;
    RetrieveStrategy* reward_strategy_;



 public:
    SumoClient(
        const string& path_to_sumo,
        const string& net,
        const string& route,
        const string& addition,
        const int yellow_time,
        const int random_seed
    ) : path_to_sumo_(path_to_sumo),
        net_(net),
        route_(route),
        addition_(addition),
        yellow_time_(yellow_time),
        random_seed_(random_seed) {
            ProcessSimulation();
            ProcessTrafficLights();
            SetStrategies();
        }

    void ProcessSimulation() {
        sumo_cmd_ = {
            path_to_sumo_,
            string("-n"),
            net_,
            string("-r"),
            route_,
            string("-a"),
            addition_,
            string("--max-depart-delay"),
            kMaxDepartDelay,
            string("--waiting-time-memory"),
            kWaitingTimeMemory,
            string("--time-to-teleport"),
            kTimeToTeleport,
            string("--no-warnings"),
            kNoWarnings,
            string("--seed"),
            std::to_string(random_seed_)
        };
        Simulation::start(sumo_cmd_);
    }

    void ProcessTrafficLights() {
        vector<std::string> tls_ids = TrafficLight::getIDList();
        std::for_each(tls_ids.begin(), tls_ids.end(), [this](const string& id) {
            traffic_lights_.emplace_back(std::make_unique<TrafficLight>(id, yellow_time_));
        });
    }

    void SetStrategies() {
        observation_strategy_ = new RetrieveObservation();
        reward_strategy_ = new RetrieveReward();
    }

    // auto RetrieveObservation() {
    //     // Implementation...
    // }
};


#endif // SUMOCLIENT_H
