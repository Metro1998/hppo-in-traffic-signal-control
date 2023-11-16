#include "trafficlight.h"
#include <iostream>
#include <functional>
#include <utility>

namespace libsumo {


TrafficLightImp::TrafficLightImp(const std::string& tls_id , int yellow_time) : tls_id_(tls_id), yellow_time_(yellow_time) {
    stage_pre_ = -1;
    stage_cur_ = -1;
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

TrafficLightImp::~TrafficLightImp() {
    // Clean up resources if neededcomdadada
}

int TrafficLightImp::Check() {
    // Check if the schedule is empty first. If it is, throw an exception.
    if (schedule_.empty()) {
        throw std::runtime_error("Error: Schedule is empty.");
    }

    // Check whether the yellow stage is over and automatically extend the green light.
    // | 0 | 0 | 0 | 16 |  --->  | 0 | 0 | 0 | 0 | ... | 0 | -1 |
    //                                       {     16X     } where -1 indicates that the agent should get a new action
    if (schedule_.front() > 0) {
        TrafficLight::setPhase(tls_id_, stage_pre_);
        for (int i = 0; i < schedule_.front(); ++i) {
            schedule_.push_back(0);
        }
        schedule_.pop_front();
        schedule_.push_back(-1);
    }

    return schedule_.front();
}

void TrafficLightImp::SchedulePop() {
    schedule_.pop_front();
}

void TrafficLightImp::SetStageDuration(const int stage, const int duration) {
    if (stage_pre_ != -1 && stage_pre_ != stage_cur_){
        int statge = mapping_[stage_pre_][stage_cur_];
        TrafficLight::setPhase(tls_id_, stage);
        for (int i = 0; i < yellow_time_; ++i){
            schedule_.push_back(0);
        }
    }

    stage_pre_ = stage_cur_;
    schedule_.push_back(duration);
    
}

// Define a template function that accepts a function pointer and a variable number of arguments
template<typename Func, typename... Args>
auto TrafficLightImp::retrieve(Func f, Args... args) -> decltype(f(std::forward<Args>(args)...)) {
    // Use std::invoke to call the function and std::forward to keep the arguments in perfect condition
    return std::invoke(f, std::forward<Args>(args)...);
}

void TrafficLightImp::UpdateLanes() {
    auto all_conns = TrafficLight::getControlledLinks(this->tls_id_);
    for_each(all_conns.begin(), all_conns.end(), [this](const auto& conn){
        // todo
        in_lanes_.emplace_back(conn.first.first);
        out_lanes_.emplace_back(conn.first.second);
    });

    RemoveElements(in_lanes_);
    RemoveElements(out_lanes_);
}


void TrafficLightImp::RemoveElements(std::vector<int>& lanes) {
        for (int i = lanes.size() - 1; i >= 0; --i) {
            if (i % 3 == 0 || i % 2 == 0) {
                lanes.erase(lanes.begin() + i);
            }
        }
    }



// You might also need to implement other functions or utilities that are necessary
// for the operation of your TrafficLightImp class

} // namespace libsumo
