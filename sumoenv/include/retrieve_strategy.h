#ifndef RETRIEVE_STRATEGY_H
#define RETRIEVE_STRATEGY_H

#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>
#include <variant
#include <iostream>
#include "libsumo/libsumo.h"

using TrafficLight = libsumo::TrafficLight;
using Lane = libsumo::Lane;
using string = std::string;
template <typename T>
using vector = std::vector<T>
using ContainerVariant = std::variant<
    std::vector<std::vector<int>>,
    std::vector<std:
    std::vector<std::pair<float, float>>
>;

class RetrieveStrategy {
  public:
    RetrieveStrategy();
    virtual const std::unordered_map<string, ContainerVariant>& Retrieve() = 0;
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
    ObservationStrategy();
    const std::unordered_map<string, ContainerVariant>& Retrieve() override;
  
  private:
    std::unordered_map<string, ContainerVariant> observation;
};

class RewardStrategy : public RetrieveStrategy {
  public:
    RewardStrategy();
    const std::unordered_map<string, ContainerVariant>& Retrieve() override;
  
  private:
    std::unordered_map<string, ContainerVariant> reward;
};

#endif // RETRIEVE_STRATEGY_H
