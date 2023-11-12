#ifndef TRAFFICLIGHT_H
#define TRAFFICLIGHT_H

#include <string>
#include <vector>
#include <deque>
#include <libsumo/libsumo.h>

namespace libsumo {
namespace trafficlight{

class TrafficLightImp
{
public:
    TrafficLightImp(std::string tlsID_, int yellowTime_);
    ~TrafficLightImp();

    void subscribe();
    void schedulePop();
    void setStageDuration(int stage, int duration);

private:
    int stagePre;
    int stageCur;
    int yellowTime;
    std::string tlsID;
    std::deque<int> schedule;
    std::vector<std::string> inlanes;
    std::vector<std::string> outlanes;

};
}
}
































#endif // TRAFFICLIGHT_H