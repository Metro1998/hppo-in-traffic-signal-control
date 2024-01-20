#include "sumo_client.h"


int main() {
    SumoClient sumo_client_ = SumoClient(
        "/usr/bin/sumo",
        "../4_4.net.xml",
        "../4_4_high.rou.xml",
        "../4_4.add.xml",
        3,
        42,
        3600.0
    );
    
    sumo_client_.Retrieve();
    sumo_client_.TempTest();
    
    // sumo_client_.RetrieveReward();

    // const auto& obs = sumo_client_.RetrieveObservation();

    // const auto& res = std::get<std::vector<std::vector<double>>>(obs.at("lane_length"));
    // std::cout << res.size() << res[0].size() << std::endl;

    // std::cout << obs.size() << obs[0].size() << std::endl;

    // std::cout << res[0][0] << std::endl;
    // g++ -o main sumoenv/main.cpp sumoenv/traffic_light.cpp -L ~/software/sumo/bin/ -l sumocpp


    std::cout << "success" << std::endl;
    
}