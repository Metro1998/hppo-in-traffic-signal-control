import numpy as np
import copy
import traci
from gym import spaces
from environments.BaseIntersection import BaseIntersectionEnv_v0


class DiscreteIntersectionEnv_v1(BaseIntersectionEnv_v0):
    """
    Description:
        A traffic signal control simulator environment for an isolated intersection.
        We supposed that there is no concept of cycle in the signal control.
        You should decide which stage to be executed in the future and its duration is fixed, i.e. 20s.
        Of course you may execute one specific stage repeatedly before the others are executed.
        It's a RL problem with discrete actions space actually.

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

    Reward:
        Number of departed vehicles between two stages.

    Starting State:
        Initialization according to sumo.

    Episode Termination:
        Episode length is greater than SIMULATION_STEPS(3600 in default, for an hour).
    """

    def __init__(self, sumocfg):
        super(DiscreteIntersectionEnv_v1, self).__init__(sumocfg)

        self.stage_duration = 25
        self.action_space = spaces.Discrete(self.stage_num)

    def step(self, action: np.float64):
        """
        :param action: the stage next
        :return: next_state, reward, done, info
        """
        stage_next = int(action)

        # SmartWolfie is a traffic signal control program defined in Intersection.add.xml.
        # We achieve Discrete actions space control through deciding its stage.
        # There is possibility that stage next period is same with the stage right now.
        # Otherwise there is a yellow between two different stages(details in self.stage_transformer).

        if not self.stage_pre is None and stage_next != self.stage_pre:
            yellow = self.stage_transformer[self.stage_pre][stage_next]
            traci.trafficlight.setPhase('SmartWolfie', yellow)
            for t in range(self.yellow):
                traci.simulationStep()
                self.episode_steps += 1

            # red = self.stage_transformer_red[self.stage_pre][stage_next]
            # traci.trafficlight.setPhase('SmartWolfie', red)
            # for _ in range(self.all_red):
            #     traci.simulationStep()
            #     self.episode_steps += 1

        traci.trafficlight.setPhase('SmartWolfie', stage_next)
        for t in range(self.stage_duration):
            traci.simulationStep()
            self.episode_steps += 1

        raw = self.retrieve_raw()
        ### states
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

