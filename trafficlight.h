#ifndef TRAFFICLIGHT_H
#define TRAFFICLIGHT_H

#include <string>
#include <vector>
#include <deque>
#include <libsumo/libsumo.h>

namespace libsumo {

class TrafficLightImp {
 public:
    TrafficLightImp(const std::string& tls_id, int yellow_time);
    ~TrafficLightImp();

    int Check();
    void SchedulePop();
    void SetStageDuration(int stage, int duration);

    template<typename Func, typename... Args>
    auto retrieve(Func f, Args... args) -> decltype(f(std::forward<Args>(args)...));
    void UpdateLanes();

 private:
    int stage_pre_;
    int stage_cur_;
    int yellow_time_;
    std::string tls_id_;
    std::deque<int> schedule_;
    std::vector<std::vector<int>> mapping_;

    std::vector<int> in_lanes_;
    std::vector<int> out_lanes_;

    void RemoveElements(std::vector<int>& lanes);
};

} // namespace libsumo

#endif // TRAFFICLIGHT_H