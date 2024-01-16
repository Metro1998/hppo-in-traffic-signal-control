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
class SumoClient { 
 private:
    const string path_to_sumo_;
    const string net_;
    const string route_;
    const string addition_;
    const int yellow_time_;
    const int random_seed_;
    const double end_time_; 

    static const string kMaxDepartDelay;
    static const string kWaitingTimeMemory;
    static const string kTimeToTeleport;
    static const string kNoWarnings;

    vector<string> sumo_cmd_;
    vector<std::unique_ptr<TrafficLightImp>> traffic_lights_;
    std::unique_ptr<RetrieveStrategy> observation_strategy_;
    std::unique_ptr<RetrieveStrategy> reward_strategy_;

    std::unordered_map<string, ContainerVariant> observation_;
    std::unordered_map<string, ContainerVariant> reward_;

 public:
    SumoClient(
        const string& path_to_sumo,
        const string& net,
        const string& route,
        const string& addition,
        int yellow_time,
        int random_seed,
        double end_time
    );

    void SetSimulation();
    void SetTrafficLights();
    void SetStrategies(); // 这些成员函数的归属 也是一个值得讨论的问题
    const std::unordered_map<string, ContainerVariant>& RetrieveObservation(); // 这里会有好几层引用传递的问题
    const std::unordered_map<string, ContainerVariant>& RetrieveReward();

    void reset();
    void step();
    void close();

    //当这些最基础的东西成熟之后，至少现在有点零乱
    //sumoenv？


};

#endif // SUMOCLIENT_H
