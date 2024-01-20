#include "sumo_client.h"

const string SumoClient::kMaxDepartDelay = "-1";
const string SumoClient::kWaitingTimeMemory = "1000";
const string SumoClient::kTimeToTeleport = "-1";
const string SumoClient::kNoWarnings = "true";


SumoClient::SumoClient(
    const string& path_to_sumo,
    const string& net,
    const string& route,
    const string& addition,
    const int yellow_time,
    const int random_seed,
    const double end_time
) : path_to_sumo_(path_to_sumo),
    net_(net),
    route_(route),
    addition_(addition),
    yellow_time_(yellow_time),
    random_seed_(random_seed),
    end_time_(end_time),
    sumo_cmd_({
        path_to_sumo,
        "--net-file", net,
        "--route-files", route,
        "--additional-files", addition,
        "--max-depart-delay", kMaxDepartDelay,
        "--waiting-time-memory", kWaitingTimeMemory,
        "--time-to-teleport", kTimeToTeleport,
        "--no-warnings", kNoWarnings,
        "--seed", std::to_string(random_seed),
        "--end", std::to_string(end_time)
    })
    {   
        auto res = Simulation::start(sumo_cmd_);
        SetTrafficLights();
        SetStrategies();
    }

void SumoClient::SetTrafficLights() {
    vector<std::string> tls_ids = TrafficLight::getIDList();
    std::for_each(tls_ids.begin(), tls_ids.end(), [this](const string& id) {
        traffic_lights_.emplace_back(std::make_unique<TrafficLightImp>(id, yellow_time_));
    });
}

void SumoClient::SetStrategies() {
    retrieve_strategy_imp_ = std::make_unique<RetrieveStrategyImp>();
}

const std::unordered_map<string, ContainerVariant>& SumoClient::Retrieve() {
    retrieve_strategy_imp_->Retrieve(this->context_);
    return context_;
}

void SumoClient::Reset() {
    // random seed?
    Simulation::close();
    auto res = Simulation::start(sumo_cmd_);
    std::cout << res.second << std::endl;
}

void SumoClient::Step(const vector<std::pair<int, int>>& action) {
    for (int i = 0; i < action.size(); ++i) {
        traffic_lights_[i]->SetStageDuration(action[i].first, action[i].second);
    }

    vector<int> checks;

    while (true)
    {
        Simulation::step();
        for (const auto& tl : traffic_lights_) {
            checks.push_back(-tl->Check());
            tl->Pop();
        }

        if (std::find(checks.begin(), checks.end(), 1) != checks.end() || Simulation::getTime() >= Simulation::getEndTime()) {
            vector<int> agents_to_update = std::move(checks);
            break;
        }
    }
    retrieve_strategy_imp_->Retrieve(this->context_);

    State state = Allocate();
    return;
}


void SumoClient::TempTest() {
    Simulation::step();
    Simulation::step();
    double res = Simulation::getEndTime();
    std::cout << "time:" << Simulation::getTime() << std::endl;
    std::cout << "cur_time:" << Simulation::getCurrentTime() << std::endl;
    std::cout << Simulation::getEndTime() << std::endl;
    // gdb debug
    // time
    // reset step close
    
}


 while True:

            self.sumo.simulationStep()
            checks = [tl.check() for tl in self.tls]
            [tl.pop() for tl in self.tls]

            self._step += 1
            if self._step >= self.max_step_episode:
                self.terminated = True

            if -1 in checks or self.terminated:
                self.agents_to_update = -np.array(checks, dtype=np.int64)
                break

        [tl.get_subscription_result() for tl in self.tls]

        if self.observation_pattern == 'queue':
            observation = np.array([tl.retrieve_queue() for tl in self.tls]).flatten()
        elif self.observation_pattern == 'pressure':
            observation = np.array([tl.retrieve_pressure() for tl in self.tls]).flatten()
        else:
            raise NotImplementedError
        
        reward = np.array([self.tls[i].retrieve_reward() if self.agents_to_update[i] else float('inf') for i in range(self.num_agent)])
        
        # global reward is for CTDE architecture e.g., MAPPO
        for i in range(self.num_agent):
            if self.agents_to_update[i]:
                self.queue_cur[i] = np.array([tl.retrieve_queue() for tl in self.tls]).sum()
                self.global_reward[i] = self.queue_old[i] - self.queue_cur[i]
                self.queue_old[i] = self.queue_cur[i]
                
        left_time = np.array([0 if self.agents_to_update[i] == 1 else tl.retrieve_left_time() for i, tl in enumerate(self.tls)])

        info = {'agents_to_update': self.agents_to_update, 'terminated': self.terminated, 'queue': np.array([sum(tl.retrieve_queue()) for tl in self.tls]).sum(),
                'left_time': left_time, 'global_reward': self.global_reward}

        return observation, reward, False, False, info

