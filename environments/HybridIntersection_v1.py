import numpy as np
import copy
import traci
from gym import spaces
from environments.BaseIntersection import BaseIntersectionEnv_v0


class HybridIntersectionEnv_v1(BaseIntersectionEnv_v0):
    """
    Description:
        A traffic signal control simulator environment for an isolated intersection.
        We supposed that there is no fixed phase cycle in the signal control.
        You should decide the stage next and its duration
        It's a RL problem with hybrid action space actually.

    Observation:
        Type: Box(8)
        Num  Observation                   Min      Max
        0    Phase_0  queue                 0.       100.
                            ...
        7    Phase_7  queue                 0.       100.

    Actions:
        Type: Discrete(8)
        Num   Action
        0     NS_straight
        1     EW_straight
        2     NS_left
        3     EW_left
        4     N_straight_left
        5     E_straight_left
        6     S_straight_left
        7     W_straight_left

        plus

        Type: Box(1)
        Num   Action                        Min      Max
        0     The duration of phase_next     5       50

    Reward:
        Number of departed vehicles between two phased.

    Starting State:
        Initialization according to sumo.

    Episode Termination:
        Episode length is greater than SIMULATION_STEPS(3600 in default, for an hour).
    """

    def __init__(self, sumocfg):
        super(HybridIntersectionEnv_v1, self).__init__(sumocfg)

        self.action_high = 45.
        self.action_low = 10.
        self.action_space = spaces.Tuple((
            spaces.Discrete(8),
            spaces.Box(low=np.array([self.action_low] * 8), high=np.array([self.action_high] * 8), dtype=np.float64)
        ))

    def step(self, action):
        """

        :param action: tuple
        :return:
        """

        stage_next = action[0]
        action = (self.action_high + self.action_low) / 2 + (self.action_high - self.action_low) / 2 * np.tanh(action[1][stage_next])
        stage_duration = int(np.ceil(action))

        # SmartWolfie is a traffic signal control program defined in Intersection.add.xml.
        # We achieve continuous action space control through deciding its duration.

        # There is a yellow between two different stages(details in self.stage_transformer).

        if not self.stage_pre is None and stage_next != self.stage_pre:
            yellow = self.stage_transformer[self.stage_pre][stage_next]
            traci.trafficlight.setPhase('SmartWolfie', yellow)
            for _ in range(self.yellow):
                traci.simulationStep()
                self.episode_steps += 1

        traci.trafficlight.setPhase('SmartWolfie', stage_next)
        for __ in range(stage_duration):
            traci.simulationStep()
            self.episode_steps += 1
        ### raw
        raw = self.retrieve_raw()
        
        ### state
        state = self.retrieve_state(raw)
        
        ### reward
        vehicle_now, waiting_time = self.retrieve_reward(raw)
        # the number of departed vehicles
        reward = len(set(self.vehicle_pre)) - len(set(vehicle_now))
        
        ### done
        if self.episode_steps > self.simulation_steps:
            done = 1
        else:
            done = 0

        ### info
        info = self.retrieve_info(raw)

        self.stage_pre = copy.deepcopy(stage_next)
        self.vehicle_pre = copy.deepcopy(vehicle_now)

        return state, reward, done, info


