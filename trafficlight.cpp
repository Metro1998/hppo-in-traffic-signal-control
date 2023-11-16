#include "trafficlight.h"
#include <iostream>
#include <functional>
#include <utility>

namespace libsumo {

// Constructor
TrafficLightImp::TrafficLightImp(const std::string& tls_id , int yellow_time) 
    : tls_id_(tls_id), yellow_time_(yellow_time), stage_pre_(-1) {
    mapping_ = {
        {-1, 8, 8, 8, 9, 8, 10, 8},
        {11, -1, 11, 11, 11, 12, 11, 13},
        {14, 14, -1, 14, 15, 14, 16, 14},
        {17, 17, 17, -1, 17, 18, 17, 19},
        {20, 22, 21, 22, -1, 22, 22, 22},
        {23, 24, 23, 25, 23, -1, 23, 23},
        {26, 27, 28, 27, 27, 27, -1, 27},
        {29, 30, 29, 31, 29, 29, 29, -1}
    };
}

// Destructor
TrafficLightImp::~TrafficLightImp() {
    // Clean up resources if needed
}

// Check method
int TrafficLightImp::Check() {
    if (schedule_.empty()) {
        throw std::runtime_error("Error: Schedule is empty.");
    }
    ExtendGreenLight();
    return schedule_.front();
}

// SchedulePop method
void TrafficLightImp::SchedulePop() {
    schedule_.pop_front();
}

// SetStageDuration method
void TrafficLightImp::SetStageDuration(const int stage, const int duration) {
    if (stage_pre_ != -1 && stage_pre_ != stage){
        int yellow_stage = mapping_[stage_pre_][stage];
        TrafficLight::setPhase(tls_id_, yellow_stage);
        std::fill_n(std::back_inserter(schedule_), yellow_time_, 0);
    }
    stage_pre_ = stage;
    schedule_.push_back(duration);
}

// Template retrieve function
template<typename Func, typename... Args>
auto TrafficLightImp::retrieve(Func f, Args... args) -> decltype(f(std::forward<Args>(args)...)) {
    return std::invoke(f, std::forward<Args>(args)...);
}

// UpdateLanes method
void TrafficLightImp::UpdateLanes() {
    auto all_conns = TrafficLight::getControlledLinks(this->tls_id_);
    for_each(all_conns.begin(), all_conns.end(), [this](const auto& conn){
        in_lanes_.emplace_back(conn.first.first);
        out_lanes_.emplace_back(conn.first.second);
    });
    RemoveElements(in_lanes_);
    RemoveElements(out_lanes_);
}

// ExtendGreenLight method
void TrafficLightImp::ExtendGreenLight() {
    if (schedule_.front() > 0) {
        TrafficLight::setPhase(tls_id_, stage_pre_);
        schedule_.insert(schedule_.end(), schedule_.front(), 0);
        schedule_.pop_front();
        schedule_.push_back(-1);
    }
}

// RemoveElements method
void TrafficLightImp::RemoveElements(std::vector<int>& lanes) {
    lanes.erase(std::remove_if(lanes.begin(), lanes.end(),
                               [i = 0](const auto&) mutable { return i++ % 3 == 0 || i % 2 == 0; }),
                lanes.end());
}

} // namespace libsumo