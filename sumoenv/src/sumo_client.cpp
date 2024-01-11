#include "sumo_client.h"

SumoClient::SumoClient(
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
        SetSimulation();
        SetTrafficLights();
        SetStrategies();
    }

void SumoClient::SetSimulation() {
    Simulation::close();
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
    auto res = Simulation::start(sumo_cmd_);
    std::cout << res.second << std::endl;
}

void SumoClient::SetTrafficLights() {
    vector<std::string> tls_ids = TrafficLight::getIDList();
    std::for_each(tls_ids.begin(), tls_ids.end(), [this](const string& id) {
        traffic_lights_.emplace_back(std::make_unique<TrafficLightImp>(id, yellow_time_));
    });
}

void SumoClient::SetStrategies() {
    observation_strategy_ = std::make_unique<ObservationStrategy>();
    reward_strategy_ = std::make_unique<RewardStrategy>();
}

void SumoClient::RetrieveObservation() {
    observation_strategy_->Retrieve(this->observation_);
    return;
}

void SumoClient::RetrieveReward() {
    reward_strategy_->Retrieve(this->reward_);
    return;
}
