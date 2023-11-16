// retrieve_strategy.h

#ifndef RETRIEVE_STRATEGY_H
#define RETRIEVE_STRATEGY_H

#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>
#include "libsumo/libsumo.h"

namespace libsumo {

using string = std::string;
template <typename T>
using vector = std::vector<T>;


class RetrieveStrategy {
  public:
    RetrieveStrategy() = default;
    decltype(auto) Retrieve() {;}
    virtual ~RetrieveStrategy() = default;

  protected:
    vector<string> tl_ids_;
    std::unordered_map<string, vector<libsumo::TraCILink>> in_lanes_map_;
    std::unordered_map<string, vector<libsumo::TraCILink>> out_lanes_map_; // to_sting?

    void ProcessTlsId() {
      tl_ids_ = TrafficLight::getIDList();
    }

    void ProcessLanes() {
      for (const auto& tl_id : tl_ids_) {
        auto all_conns = TrafficLight::getControlledLinks(tl_id);
        for_each(all_conns.begin(), all_conns.end(), [this, tl_id](const auto& conn){
          in_lanes_map_[tl_id].emplace_back(conn[1]);
          out_lanes_map_[tl_id].emplace_back(conn[0]);
        });
        RemoveElements(in_lanes_map_[tl_id]);
        RemoveElements(out_lanes_map_[tl_id]);
      } 
    }
    
    void RemoveElements(vector<libsumo::TraCILink>& lanes) {
    lanes.erase(std::remove_if(lanes.begin(), lanes.end(),
                               [i = 0](const auto&) mutable { return i++ % 3 == 0 || i % 2 == 0; }),
                lanes.end());
    }
    
    // template<typename Func, typename... Args>
    // auto MetaRetrieve(Func f, Args... args) -> decltype(f(std::forward<Args>(args)...)) {
    //     return std::invoke(f, std::forward<Args>(args)...);
    // }
};


class RetrieveObservation : public RetrieveStrategy {
  public:
    RetrieveObservation() : RetrieveStrategy() {
      // Process the data you need for calculating observations and rewards here.
        ProcessTlsId();
        ProcessLanes();
    }

    decltype(auto) Retrieve() {
      // Rewrite the Retrieve method to implement your own reward design or state representation.
      // Here we use 'hide' ranther than 'override' from virtual function to support decltype(auto) return type.
      vector<vector<int>> observation;

      for (const string& tl_id : tl_ids_) {
        vector<int> lane_vehicles_for_tl;
        for (const auto& lane_id : in_lanes_map_[tl_id]) {
          int lane_vehicles = Lane::getLastStepHaltingNumber(lane_id);
          lane_vehicles_for_tl.push_back(lane_vehicles);
        }
        observation.push_back(lane_vehicles_for_tl);
      }
      return observation;
    }
};


class RetrieveReward : public RetrieveStrategy {
  public:
    RetrieveReward() : RetrieveStrategy() {
      // Process the data you need for calculating observations and rewards here.
        ProcessTlsId();
        ProcessLanes();
    }

    decltype(auto) Retrieve() {
      // Rewrite the Retrieve method to implement your own reward design or state representation.
      // Here we use 'hide' ranther than 'override' from virtual function to support decltype(auto) return type.
      vector<int> reward;

      for (const string& tl_id : tl_ids_) {
        int vehicles_for_tl = 0;
        for (const auto& lane_id : in_lanes_map_[tl_id]) {
          vehicles_for_tl += Lane::getLastStepHaltingNumber(lane_id);
        }
        reward.push_back(vehicles_for_tl);
      }
      return reward;
    }
};













}














#endif // RETRIEVE_STRATEGY_H