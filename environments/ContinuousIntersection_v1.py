import numpy as np
import copy
import traci
from gym import spaces
from environments.BaseIntersection import BaseIntersectionEnv_v0


class ContinuousIntersectionEnv_v1(BaseIntersectionEnv_v0):
    """
    Description:
        A traffic signal control simulator environment for an isolated intersection.
        We supposed that there is a fixed phase cycle in the signal control.
        You should decide the duration of phase next *(when there is a fixed cycle we prefer 'phase' to 'stage')
        It's a RL problem with continuous action space actually.

    Observation:
        Type: Box(8)
        Num  Observation                   Min      Max
        0    Phase_0  queue                 0.       100.
                            ...
        7    Phase_7  queue                 0.       100.

    Actions:
        Type: Box(1)
        Num   Action                        Min      Max
        0     The duration of phase_next     10       30

    Reward:
        Number of departed vehicles between two phased.

    Starting State:
        Initialization according to sumo.

    Episode Termination:
        Episode length is greater than SIMULATION_STEPS(3600 in default, for an hour).
    """

    def __init__(self, sumocfg):
        super(ContinuousIntersectionEnv_v1, self).__init__(sumocfg)

        # consistent with phases defined in the add.xml
        # [0, 2, 1, 3] means north_south_through, north_south_left, east_west_through, east_west_left
        self.stage = np.array([4, 5, 6, 7])

        self.action_high = 50.
        self.action_low = 10.
        self.action_space = spaces.Box(low=np.array([self.action_low] * len(self.stage)), high=np.array([self.action_high] * len(self.stage)),
                                       dtype=np.float64)

    def reset(self):
        """
        Connect with the sumo instance, could be multiprocess.
        :return: number of vehicles w.r.t eight phases
        """

        path = 'envs/sumo_files/{}.sumocfg'.format(self.sumocfg)

        # create instances
        traci.start(['sumo', '-c', path], label='sim1')
        self.episode_steps = 0
        raw = self.retrieve_raw()
        state = self.retrieve_state(raw)
        done = 0

        self.stage_ptr = 0
        self.stage_pre = None
        self.vehicle_pre = np.array([])

        return state, done

    def step(self, action):
        """
        Implemantation of step in env.

        :param action: array, (4, )
        :return:
        """
        stage_next = self.stage[self.stage_ptr]
        action = (self.action_high + self.action_low) / 2 + (self.action_high - self.action_low) / 2 * np.tanh(action[self.stage_ptr])
        stage_duration = int(np.ceil(action))

        # SmartWolfie is a traffic signal control program defined in Intersection.add.xml.
        # We achieve continuous action space control through deciding its duration.

        # There is a yellow between two different stages(details in self.stage_transformer).

        if not self.stage_pre is None:
            yellow = self.stage_transformer[self.stage_pre][stage_next]
            traci.trafficlight.setPhase('SmartWolfie', yellow)
            for _ in range(self.yellow):
                traci.simulationStep()
                self.episode_steps += 1

            # red = self.stage_transformer_red[self.stage_pre][stage_next]
            # traci.trafficlight.setPhase('SmartWolfie', red)
            # for _ in range(self.all_red):
            #     traci.simulationStep()
            #     self.episode_steps += 1

        traci.trafficlight.setPhase('SmartWolfie', stage_next)
        for _ in range(stage_duration):
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
        self.stage_ptr = (self.stage_ptr + 1) % len(self.stage)
        self.vehicle_pre = copy.deepcopy(vehicle_now)

        return state, reward, done, info

