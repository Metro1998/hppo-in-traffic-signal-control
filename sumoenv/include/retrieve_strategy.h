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
using Vehicle = libsumo::Vehicle;
using string = std::string;
template <typename T>
using vector = std::vector<T>;
using ContainerVariant = std::variant<
    vector<vector<int>>,
    vector<vector<double>>,
    vector<vector<vector<double>>>,
    vector<int>,
    vector<std::pair<float, float>>
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
};

class RetrieveStrategyImp : public RetrieveStrategy {
  public:
    RetrieveStrategyImp() = default;
    void Retrieve(std::unordered_map<string, ContainerVariant>& context) override;
};

// class RewardStrategy : public RetrieveStrategy {
//   public:
//     RewardStrategy() = default;
//     void Retrieve(std::unordered_map<string, ContainerVariant>& context) override;
  
// };

#endif // RETRIEVE_STRATEGY_H
