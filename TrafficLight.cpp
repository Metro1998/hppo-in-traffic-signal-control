#include "TrafficLight.h"
#include <iostream>
#include <functional>
#include <utility>

namespace libsumo {


TrafficLightImp::TrafficLightImp(std::string tlsID_, int yellowTime_) : tlsID(tlsID_), yellowTime(yellowTime_) {
    stagePre = -1;
    stageCur = -1;
    mapping = {
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

int TrafficLightImp::check() {
    // Check if the schedule is empty first. If it is, throw an exception.
    if (schedule.empty()) {
        throw std::runtime_error("Error: Schedule is empty.");
    }

    // Check whether the yellow stage is over and automatically extend the green light.
    // | 0 | 0 | 0 | 16 |  --->  | 0 | 0 | 0 | 0 | ... | 0 | -1 |
    //                                       {     16X     } where -1 indicates that the agent should get a new action
    if (schedule.front() > 0) {
        TrafficLight::setPhase(tlsID, stagePre);
        for (int i = 0; i < schedule.front(); ++i) {
            schedule.push_back(0);
        }
        schedule.pop_front();
        schedule.push_back(-1);
    }

    return schedule.front();
}

void TrafficLightImp::schedulePop() {
    schedule.pop_front();
}

void TrafficLightImp::setStageDuration(const int stage, const int duration) {
    if (stagePre != -1 && stagePre != stageCur){
        int statge = mapping[stagePre][stageCur];
        TrafficLight::setPhase(tlsID, stage);
        for (int i = 0; i < yellowTime; ++i){
            schedule.push_back(0);
        }
    }

    stagePre = stageCur;
    schedule.push_back(duration);
    
}

// Define a template function that accepts a function pointer and a variable number of arguments
template<typename Func, typename... Args>
auto TrafficLightImp::retrieve(Func f, Args... args) -> decltype(f(std::forward<Args>(args)...)) {
    // Use std::invoke to call the function and std::forward to keep the arguments in perfect condition
    return std::invoke(f, std::forward<Args>(args)...);
}


// You might also need to implement other functions or utilities that are necessary
// for the operation of your TrafficLightImp class

} // namespace libsumo