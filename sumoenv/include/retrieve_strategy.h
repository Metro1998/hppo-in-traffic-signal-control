#ifndef RETRIEVE_STRATEGY_H
#define RETRIEVE_STRATEGY_H


#include <string>
#include <vector>
#include <variant>
#include <iostream>
#include <algorithm>
#include <unordered_map>

#include "libsumo/libsumo.h"

using TrafficLight = libsumo::TrafficLight;
using Lane = libsumo::Lane;
using string = std::string;
template <typename T>
using vector = std::vector<T>;
using ContainerVariant = std::variant<
    std::vector<std::vector<int>>,
    std::vector<float>,
    std::vector<std::pair<float, float>>
>;

class RetrieveStrategy {
  public:
    RetrieveStrategy();
    virtual void Retrieve(std::unordered_map<string, ContainerVariant>& context) = 0;
    virtual ~RetrieveStrategy();

  protected:
    vector<string> tl_ids_;
    std::unordered_map<string, vector<string>> in_lanes_map_;

    void ProcessTlsId();
    void ProcessLanes();
    void RemoveElements(vector<string>& lanes);
    void RemoveElements_(vector<string>& lanes);
};

class ObservationStrategy : public RetrieveStrategy {
  public:
    ObservationStrategy() = default;
    void Retrieve(std::unordered_map<string, ContainerVariant>& context) override;
    
};

class RewardStrategy : public RetrieveStrategy {
  public:
    RewardStrategy() = default;
    void Retrieve(std::unordered_map<string, ContainerVariant>& context) override;
  
};

#endif // RETRIEVE_STRATEGY_H
