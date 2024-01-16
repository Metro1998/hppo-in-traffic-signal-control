#include "sumo_client.h"

const string SumoClient::kMaxDepartDelay = "-1";
const string SumoClient::kWaitingTimeMemory = "1000";
const string SumoClient::kTimeToTeleport = "-1";
const string SumoClient::kNoWarnings = "true";


SumoClient::SumoClient(
    const string& path_to_sumo,
    const string& net,
    const string& route,
    const string& addition,
    const int yellow_time,
    const int random_seed,
    const double end_time
) : path_to_sumo_(path_to_sumo),
    net_(net),
    route_(route),
    addition_(addition),
    yellow_time_(yellow_time),
    random_seed_(random_seed),
    end_time_(end_time) {
        SetSimulation();
        SetTrafficLights();
        SetStrategies();
    }

void SumoClient::SetSimulation() {
    Simulation::close();
    sumo_cmd_ = {
        path_to_sumo_,
        string("-net-file"),
        net_,
        string("-route-files"),
        route_,
        string("-additional-files"),
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
        std::to_string(random_seed_),
        string("--end"),
        std::to_string(end_time_)
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

const std::unordered_map<string, ContainerVariant>& SumoClient::RetrieveObservation() {
    observation_strategy_->Retrieve(this->observation_);
    return observation_;
}

const std::unordered_map<string, ContainerVariant>& SumoClient::RetrieveReward() {
    reward_strategy_->Retrieve(this->reward_);
    return reward_;
}
