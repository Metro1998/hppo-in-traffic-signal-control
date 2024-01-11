#include "retrieve_strategy.h"

RetrieveStrategy::RetrieveStrategy() {
    ProcessTlsId();
    ProcessLanes();
};
RetrieveStrategy::~RetrieveStrategy() = default;

void RetrieveStrategy::ProcessTlsId() {
    tl_ids_ = TrafficLight::getIDList();
}

void RetrieveStrategy::ProcessLanes() {
    for (const auto& tl_id : tl_ids_) {
        in_lanes_map_[tl_id] = TrafficLight::getControlledLanes(tl_id);
        RemoveElements(in_lanes_map_[tl_id]);
        RemoveElements_(in_lanes_map_[tl_id]);
    }
}

void RetrieveStrategy::RemoveElements(vector<string>& lanes) {
    lanes.erase(std::remove_if(lanes.begin(), lanes.end(),
                               [i = 0](const auto&) mutable {
                                   bool shouldErase = i % 3 == 1 || i % 3 == 2; 
                                   i++;
                                   return shouldErase; }),
                lanes.end());

    lanes.erase(std::remove_if(lanes.begin(), lanes.end(),
                               [i = 0](const auto&) mutable { return i++ % 3 == 2;}),
                lanes.end());
}

void RetrieveStrategy::RemoveElements_(vector<string>& lanes) {
    // 这个函数的实现似乎是空的，所以暂时留空。
}

void ObservationStrategy::Retrieve(std::unordered_map<string, ContainerVariant>& context) {
    vector<vector<int>> queue_length;

    for (const string& tl_id : tl_ids_) {
        vector<int> lane_vehicles_for_tl;
        for (const auto& lane_id : in_lanes_map_[tl_id]) {
            int lane_vehicles = Lane::getLastStepHaltingNumber(lane_id);
            lane_vehicles_for_tl.push_back(lane_vehicles);
        }
        queue_length.push_back(lane_vehicles_for_tl);
    }

    context["queue_length"] = std::move(queue_length);

    return;
}


void RewardStrategy::Retrieve(std::unordered_map<string, ContainerVariant>& context) {
    // total queue_length of one trffic light
    vector<float> queue_length_tl;

    for (const string& tl_id : tl_ids_) {
        int vehicles_for_tl = 0;
        for (const auto& lane_id : in_lanes_map_[tl_id]) {
            vehicles_for_tl += Lane::getLastStepHaltingNumber(lane_id);
        }
        queue_length_tl.push_back(vehicles_for_tl);
    }
    context["queue_length_tl"] = std::move(queue_length_tl);
    return;
}
