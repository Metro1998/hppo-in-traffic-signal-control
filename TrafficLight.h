#ifndef TRAFFICLIGHT_H
#define TRAFFICLIGHT_H

#include <string>
#include <vector>
#include <deque>
#include <libsumo/libsumo.h>

namespace libsumo {

class TrafficLightImp {
public:
    TrafficLightImp(std::string tlsID_, int yellowTime_);
    ~TrafficLightImp();

    int check();
    void schedulePop();
    void setStageDuration(int stage, int duration);
    
    template<typename Func, typename... Args>
    auto TrafficLightImp::retrieve(Func f, Args... args) -> decltype(f(std::forward<Args>(args)...));


private:
    int stagePre;
    int stageCur;
    int yellowTime;
    std::string tlsID;
    std::deque<int> schedule;
    std::vector<std::string> inlanes;
    std::vector<std::string> outlanes;
    std::vector<std::vector<int>> mapping;

};

} // namespace libsumo
































#endif // TRAFFICLIGHT_H