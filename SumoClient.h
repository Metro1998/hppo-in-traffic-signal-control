#ifndef SUOMCLIENT_H
#define SUOMCLIENT_H

#include <libsumo/libsumo.h>
#include <trafficlight.h>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>

namespace libsumo {

class SumoClient {
private:
    std::string sumo_binary;
    std::string net;
    std::string route;
    std::string addition;
    int yellow_time;
    int random_seed;

    const int max_depart_delay = -1;
    const int waiting_time_memory = 1000;
    const int time_to_teleport = -1;
    const bool no_warnings = true;

    std::vector<std::string> sumo_cmd;
    std::vector<std::unique_ptr<TrafficLightImp>> traffic_lights;

public:
    SumoClient(
        const std::string& net_,
        const std::string& route_,
        const std::string& addition_,
        int yellow_time_,
        int random_seed_
    ) : net(net_),
        route(route_),
        addition(addition_),
        yellow_time(yellow_time_),
        random_seed(random_seed_){
            prepareSimulation();
            prepareTrafficLights();
        }

    void prepareSimulation(){
        // todo
        sumo_binary = 
        sumo_cmd = {
            sumo_binary,
            std::string("-n"),
            net,
            std::string("-r"),
            route,
            std::string("-a"),
            addition,
            std::string("--max-depart-delay"),
            std::to_string(max_depart_delay),
            std::string("--waiting-time-memory"),
            std::to_string(waiting_time_memory),
            std::string("--time-to-teleport"),
            std::to_string(time_to_teleport),
            std::string("--no-warnings"),
            std::to_string(no_warnings),
            std::string("--seed"),
            std::to_string(random_seed)
        };
        Simulation::start(sumo_cmd);

    }

    void prepareTrafficLights(){
        std::vector<std::string> tlsIDs = TrafficLight::getIDList();
        std::for_each(tlsIDs.begin(), tlsIDs.end(), [this](const std::string& tlsIDs, int yellow_time){
            traffic_lights.emplace_back(std::make_unique<TrafficLightImp>(tlsIDs, yellow_time));
        });
    }





    

}






}


































#endif // SUOMCLIENT_H