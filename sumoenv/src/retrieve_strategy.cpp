#include <string>

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
    }
}

void RetrieveStrategy::RemoveElements(vector<string>& lanes) {
    vector<string> temp;

    // std::cout << "before remove: " << std::endl;
    // for (const auto& lane : lanes) {
    //     std::cout << lane << std::endl;
    // }

    for (size_t i = 0; i < lanes.size(); ++i) {
        if (i % 3 == 0) {
            temp.emplace_back(lanes[i]);
        }
    }
    lanes = std::move(temp);
    temp.clear();

    for (size_t i = 0; i < lanes.size(); ++i) {
        if (i % 3 != 0) {
            temp.emplace_back(lanes[i]);
        }
    }
    lanes = std::move(temp);

    // std::cout << "after remove: " << std::endl;
    // for (const auto& lane : lanes) {
    //     std::cout << lane << std::endl;
    // }

    return;
}


void ObservationStrategy::Retrieve(std::unordered_map<string, ContainerVariant>& context) {
    vector<vector<double>> lane_lengths(tl_ids_.size());
    vector<vector<int>> lane_queue_lengths(tl_ids_.size());
    vector<vector<double>> lane_max_speeds(tl_ids_.size());
    vector<vector<vector<double>>> vehicle_speeds(tl_ids_.size());
    vector<vector<vector<double>>> vehicle_positions(tl_ids_.size());
    vector<vector<vector<double>>> vehicle_acceleratoins(tl_ids_.size());

    int tl_index = 0;
    for (const string& tl_id : tl_ids_) {
        const auto& lanes = in_lanes_map_[tl_id];
        lane_lengths[tl_index].reserve(lanes.size());
        lane_queue_lengths[tl_index].reserve(lanes.size());
        lane_max_speeds[tl_index].reserve(lanes.size());
        vehicle_speeds[tl_index].reserve(lanes.size());
        vehicle_positions[tl_index].reserve(lanes.size());
        vehicle_acceleratoins[tl_index].reserve(lanes.size());

        int lane_index = 0;
        for (const string& lane_id : lanes) {
            std::cout << lanes.size() << std::endl;
            lane_lengths[tl_index].emplace_back(Lane::getLength(lane_id));
            lane_queue_lengths[tl_index].emplace_back(Lane::getLastStepHaltingNumber(lane_id));
            lane_max_speeds[tl_index].emplace_back(Lane::getMaxSpeed(lane_id));

            const auto& last_step_vehicle_ids = Lane::getLastStepVehicleIDs(lane_id);
            vehicle_speeds[tl_index][lane_index].reserve(last_step_vehicle_ids.size());
            vehicle_positions[tl_index][lane_index].reserve(last_step_vehicle_ids.size());
            for (const string& vehicle_id : last_step_vehicle_ids) {
                vehicle_speeds[tl_index][lane_index].emplace_back(Vehicle::getSpeed(vehicle_id));
                vehicle_positions[tl_index][lane_index].emplace_back(Vehicle::getLanePosition(vehicle_id));
                vehicle_acceleratoins[tl_index][lane_index].emplace_back(Vehicle::getAcceleration(vehicle_id));
            }

            ++lane_index;
        }
        ++tl_index;
    }

    context["lane_length"] = std::move(lane_lengths);
    context["lane_queue_length"] = std::move(lane_queue_lengths);
    context["lane_max_speed"] = std::move(lane_max_speeds);
    context["vehicle_speed"] = std::move(vehicle_speeds);
    context["vehicle_position"] = std::move(vehicle_positions);
    context["vehicle_acceleration"] = std::move(vehicle_acceleratoins);

    return;
}

void RewardStrategy::Retrieve(std::unordered_map<string, ContainerVariant>& context) {
    // total queue_length of one trffic light
    vector<int> intersection_queue_length;

    for (const string& tl_id : tl_ids_) {
        int vehicles_for_tl = 0;
        for (const auto& lane_id : in_lanes_map_[tl_id]) {
            vehicles_for_tl += Lane::getLastStepHaltingNumber(lane_id);
        }
        intersection_queue_length.push_back(vehicles_for_tl);
    }
    context["intersection_queue_length"] = std::move(intersection_queue_length);
    return;
}
