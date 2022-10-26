import gym
import numpy as np
import sys
import random
import traci
import traci.constants as tc
from gym import spaces


class BaseIntersectionEnv_v0(gym.Env):
    """
    Description:
        A base traffic signal control simulator environment for an isolated intersection.


    Observation:
        Implemented in its subclass, but a specific case provided.

    Actions:
        Implemented in its subclass.

    Reward:
        Implemented in its subclass, but a specific case provided.

    Starting State:
        Initialization according to sumo.

    Episode Termination:
        Episode length is greater than SIMULATION_STEPS(3600 in default, for an hour).
    """

    def __init__(self, sumocfg):
        self.sumocfg = sumocfg
        self.stage_num = 8
        self.evaluation_frequency = 5
        self.info_list = []

        # the edgeID is defined in Intersection.edg.xml.
        # as you may have different definition in your own .edg.xml, change it.
        self.edgeIDs = ['north_in', 'east_in', 'south_in', 'west_in']

        # vehicle_types will help to filter the vehicles on the same edge but have different direction.
        self.vehicle_types = ['NS_through', 'NE_left', 'EW_through', 'ES_left',
                              'SN_through', 'SW_left', 'WE_through', 'WN_left']

        # stage_transformer will help you decide which YELLOW should be inserted between different stages.
        self.stage_transformer = np.array([
            [None, 8, 8, 8, 16, 8, 17, 8],
            [9, None, 9, 9, 9, 18, 9, 19],
            [10, 10, None, 10, 20, 10, 21, 10],
            [11, 11, 11, None, 11, 22, 11, 23],
            [24, 12, 25, 12, None, 12, 12, 12],
            [13, 26, 13, 27, 13, None, 13, 13],
            [28, 14, 29, 14, 14, 14, None, 14],
            [15, 30, 15, 31, 15, 15, 15, None]
        ])

        self.stage_transformer_red = np.array([
            [None, 32, 32, 32, 33, 32, 34, 32],
            [32, None, 32, 32, 32, 37, 32, 38],
            [32, 32, None, 32, 35, 32, 36, 32],
            [32, 32, 32, None, 32, 39, 32, 40],
            [33, 32, 35, 32, None, 32, 32, 32],
            [32, 37, 32, 39, 32, None, 32, 32],
            [34, 32, 36, 32, 32, 32, None, 32],
            [32, 38, 32, 40, 32, 32, 32, None]
        ])

        self.lane_length = 240.
        self.yellow = 3
        self.all_red = 1
        self.red_stage = 32
        self.max_queuing_speed = 1  # TODO
        self.simulation_steps = 3600

        self.episode_steps = 0
        self.stage_ptr = None
        self.stage_pre = None
        self.vehicle_pre = None

        self.action_high = None
        self.action_low = None

        self.seed(1)

        observation_low = np.array([0] * self.stage_num)
        observation_high = np.array([100] * self.stage_num)
        self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float64)

        sys.path.append('D:/SUMO/tools')

        self.all_vehicles_delay_recorder = {}

    def set_action_space(self):
        """
        Implemented in its subclass.
        """

    """
    Since we are interested in the influence of different action space patterns, the definitions of
    observation space/ state/ reward are fixed. The latter ones are available to be modified if necessary 
    for your own research.
    
    """

    def reset(self):
        """
        Connect with the sumo instance, could be multiprocess.
        :return: number of vehicles w.r.t eight phases
        """
        path = 'envs/sumo_files/{}.sumocfg'.format(self.sumocfg)

        traci.start(['sumo', '-c', path], label='sim1')
        self.episode_steps = 0
        raw = self.retrieve_raw()
        state = self.retrieve_state(raw)
        done = 0

        self.stage_pre = None
        self.vehicle_pre = []

        return state, done

    def step(self, action):
        """
        :param action
        :return: next_state, reward, done, info

        Implemented in its subclass.
        """

    def retrieve_raw(self):
        """
        :return: dic to save_policy vehicles' speed and position etc. w.r.t its vehicle type
        """
        vehicle_raw_info = {_: [] for _ in self.vehicle_types}
        for edgeID in self.edgeIDs:
            traci.edge.subscribe(edgeID, (tc.LAST_STEP_VEHICLE_ID_LIST,))
            for _ in traci.edge.getSubscriptionResults(edgeID).values():
                vehicle_IDs_edge = [IDs for IDs in _]

                for ID in vehicle_IDs_edge:

                    traci.vehicle.subscribe(ID, (tc.VAR_TYPE, tc.VAR_LANEPOSITION, tc.VAR_SPEED,
                                                 tc.VAR_ACCUMULATED_WAITING_TIME, tc.VAR_TIMELOSS))
                    temp = [_ for _ in traci.vehicle.getSubscriptionResults(ID).values()]
                    temp[1] = self.lane_length - temp[1]
                    if temp[0] in self.vehicle_types:
                        vehicle_raw_info[temp[0]].append([ID, temp[1], temp[2], temp[3], temp[4]])
                        # LENGTH_LANE is the length of lane, gotten from FW_Inter.net.xml.
                        # temp[0]:str, vehicle's ID
                        # temp[1]:float, the distance between vehicle and lane's stop line.
                        # temp[2]:float, speed
                        # temp[3]:float, accumulated_waiting_time
                        # temp[4]:float, time loss
        return vehicle_raw_info

    @staticmethod
    def retrieve_state(raw):
        """
        :return:
        """

        state = np.array([len(v) for k, v in raw.items()])

        return state

    @staticmethod
    def retrieve_reward(raw):
        """
        :return:
        """

        vehicle_IDs = np.array([_[0] for k, v in raw.items() for _ in v])
        acc_waiting_time = np.array([_[3] for k, v in raw.items() for _ in v])

        return vehicle_IDs, acc_waiting_time

    def retrieve_info(self, raw):
        """
        retrieve information for evaluation
        :param raw:
        :return:
        """

        loss_time = np.mean([_[4] for k, v in raw.items() for _ in v])
        queue = len([_[2] for k, v in raw.items() for _ in v if _[2] < self.max_queuing_speed])

        return queue, loss_time

    def retrieve_info_max(self, raw):
        """
        retrieve information for evaluation
        :param raw:
        :return:
        """

        loss_time = np.max([_[4] for k, v in raw.items() for _ in v])
        queue = len([_[2] for k, v in raw.items() for _ in v if _[2] < self.max_queuing_speed])

        return queue, loss_time

    @staticmethod
    def retrieve_delay_by_IDs(IDs):
        delay = []
        for ID in IDs:
            traci.vehicle.subscribe(ID, [tc.VAR_TIMELOSS])
            temp = [_ for _ in traci.vehicle.getSubscriptionResults(ID).values()]
            delay.append(temp[4])

        return delay

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def render(self, mode='human'):
        pass

    def close(self):
        """
        :return:
        """
        traci.close()

    def evaluate(self):
        raw = self.retrieve_raw()
        info = self.retrieve_info(raw)

        return info
