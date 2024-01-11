#ifndef SUMOCLIENT_H
#define SUMOCLIENT_H

#include <vector>

#include <memory>
#include <string>
#include <variant>
#include <iostream>
#include <algorithm>
#include <unordered_map>

#include <libsumo/libsumo.h>

#include "traffic_light.h"
#include "retrieve_strategy.h"

using Simulation = libsumo::Simulation;
using string = std::string;
template <typename T>
using vector = std::vector<T>;
using ContainerVariant = std::variant<
    std::vector<std::vector<int>>,
    std::vector<float>,
    std::vector<std::pair<float, float>>
>;

class SumoClient { 
 private:
    string path_to_sumo_;
    string net_;
    string route_;
    string addition_;
    int yellow_time_;
    int random_seed_;
    std::unordered_map<string, ContainerVariant> observation_;
    std::unordered_map<string, ContainerVariant> reward_;

    const string kMaxDepartDelay = "-1";
    const string kWaitingTimeMemory = "1000";
    const string kTimeToTeleport = "-1";
    const string kNoWarnings = "true";

    vector<string> sumo_cmd_;
    vector<std::unique_ptr<TrafficLightImp>> traffic_lights_;

    std::unique_ptr<RetrieveStrategy> observation_strategy_;
    std::unique_ptr<RetrieveStrategy> reward_strategy_;

 public:
    SumoClient(
        const string& path_to_sumo,
        const string& net,
        const string& route,
        const string& addition,
        const int yellow_time,
        const int random_seed
    );

    void SetSimulation();
    void SetTrafficLights();
    void SetStrategies();
    void RetrieveObservation();
    void RetrieveReward();
};

#endif // SUMOCLIENT_H
