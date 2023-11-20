#ifndef TRAFFIC_LIGHT_H
#define TRAFFIC_LIGHT_H

#include <string>
#include <vector>
#include <deque>
#include <libsumo/libsumo.h>

class TrafficLightImp {
 public:
    TrafficLightImp(const std::string& tls_id, int yellow_time);
    ~TrafficLightImp();

    int Check();
    inline void SchedulePop();
    void SetStageDuration(int stage, int duration);

    
   //  void UpdateLanes();

 private:
    int stage_pre_;
    int yellow_time_;
    std::string tl_ids_;
    std::deque<int> schedule_;
    std::vector<std::vector<int>> mapping_;
    void ExtendGreenLight();

};


#endif // TRAFFIC_LIGHT_H