#include "TrafficLight.h"
#include <iostream>

namespace libsumo {
namespace trafficlight {

TrafficLightImp::TrafficLightImp(std::string tlsID_, int yellowTime_) : tlsID(tlsID_), yellowTime(yellowTime_) {
    stagePre = -1;
    stageCur = -1;
}

TrafficLightImp::~TrafficLightImp() {
    // Clean up resources if neededcomdadada
}

void TrafficLightImp::subscribe() {
    
}

void TrafficLightImp::schedulePop() {
    this->schedule.pop_front();
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

// You might also need to implement other functions or utilities that are necessary
// for the operation of your TrafficLightImp class

} // namespace trafficlight
} // namespace libsumo