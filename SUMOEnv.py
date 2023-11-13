import gymnasium as gym
import numpy as np
import libsumo as traci
import traci
import sumolib
import os

from gymnasium import spaces
from collections import deque
from typing import Callable, Optional, Tuple, Union, List

LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


class TrafficSignal:
    """
    This class represents a Traffic Signal controlling an intersection.
    It is responsible for retrieving information and changing the traffic phase using the Traci API.

    IMPORTANT!!! NOTE THAT
    Our reward is defined as the change in vehicle number of one specific junction.
    Our observations is defined as the pressure between the inlanes and outlanes.
    """

    def __init__(self, tl_id, yellow, sumo):

        self.id = tl_id
        self.yellow = yellow
        self.sumo = sumo

        # The schedule is responsible for the automatic timing for the incoming green stage.
        # | 0 | 0 | 0 | 16 |
        # | yellow len| when 16 is dequeued, the stage is automatically transferred to the green stage and 16 is for duration.
        self.schedule = deque()

        # Links is relative with connections defined in the rou.xml, what's more the connection definition should be
        # relative with traffic observations definition. Therefore, there is no restriction that the connection should start
        # at north and step clockwise then.
        all_lanes = self.sumo.trafficlight.getControlledLinks(self.id)
        self.in_lanes = [conn[0][0] for conn in all_lanes]
        self.out_lanes = [conn[0][1] for conn in all_lanes]
        del self.in_lanes[0::3]
        del self.in_lanes[0::2]
        del self.in_lanes[0::3]
        del self.out_lanes[0::3]
        del self.out_lanes[0::2]
        del self.out_lanes[0::3]

        self.subscribe()

        self.inlane_halting_vehicle_number = None
        self.inlane_halting_vehicle_number_old = None
        self.inlane_waiting_time = None
        self.outlane_halting_vehicle_number = None
        self.outlane_waiting_time = None
        self.stage_old = None

        self.mapping = np.array([
            [-1, 8, 8, 8, 9, 8, 10, 8],
            [11, -1, 11, 11, 11, 12, 11, 13],
            [14, 14, -1, 14, 15, 14, 16, 14],
            [17, 17, 17, -1, 17, 18, 17, 19],
            [20, 22, 21, 22, -1, 22, 22, 22],
            [23, 24, 23, 25, 23, -1, 23, 23],
            [26, 27, 28, 27, 27, 27, -1, 27],
            [29, 30, 29, 31, 29, 29, 29, -1],
        ])

    def set_stage_duration(self, stage: int, duration: int):
        """
        Call this at the beginning the of one stage, which includes the switching yellow light between two different
        green light.
        In add.xml the stage is defined as the yellow stage then next green stage, therefore the yellow stage is first
        implemented, and after self.yellow seconds, it will automatically transfer to green stage, through a schedule to
        set the incoming green stage's duration.
        :return:
        """
        if self.stage_old is not None and self.stage_old != stage:
            yellow_stage = int(self.mapping[self.stage_old][stage])
            self.sumo.trafficlight.setPhase(self.id, yellow_stage)
            for i in range(self.yellow - 1):
                self.schedule.append(0)
        self.stage_old = int(stage)
        self.schedule.append(duration)

    def check(self):
        """
        Check whether the yellow stage is over and automatically extend the green light.
        # | 0 | 0 | 0 | 16 |  --->  | 0 | 0 | 0 | 0 | ... | 0 | -1 |
        #                                       {     16X     } where -1 indicates that the agent should get a new action
        :return:
        """
        if self.schedule[0] > 0:
            self.sumo.trafficlight.setPhase(self.id, self.stage_old)
            for i in range(self.schedule[0]):
                self.schedule.append(0)
            self.schedule.popleft()
            self.schedule.append(-1)

        return self.schedule[0]

    def pop(self):
        self.schedule.popleft()

    def subscribe(self):
        """
        Pre subscribe the information we interest, so as to accelerate the information retrieval.
        See https://sumo.dlr.de/docs/TraCI.html "Performance" for more detailed explanation.
        :return:
        """

        for lane_id in self.in_lanes:
            self.sumo.lane.subscribe(lane_id, [traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER,
                                               traci.constants.VAR_WAITING_TIME])

        for lane_id in self.out_lanes:
            self.sumo.lane.subscribe(lane_id, [traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER,
                                               traci.constants.VAR_WAITING_TIME])

    def get_subscription_result(self):
        self.inlane_halting_vehicle_number = np.array(
            [list(self.sumo.lane.getSubscriptionResults(lane_id).values())[0] for lane_id in self.in_lanes])

        self.= np.array(
            [list(self.sumo.lane.getSubscriptionResults(lane_id).values())[0] for lane_id in self.out_lanes])

        self.inlane_waiting_time = np.array(
            [list(self.sumo.lane.getSubscriptionResults(lane_id).values())[1] for lane_id in self.in_lanes])

    def retrieve_reward(self):
        if not isinstance(self.inlane_halting_vehicle_number_old, np.ndarray):
            reward = -sum(self.inlane_halting_vehicle_number)
        else:
            reward = sum(self.inlane_halting_vehicle_number_old) - sum(self.inlane_halting_vehicle_number)

        self.inlane_halting_vehicle_number_old = self.inlane_halting_vehicle_number

        return reward

    def retrieve_pressure(self):
        pressure = self.inlane_halting_vehicle_number - self.outlane_halting_vehicle_number

        return pressure

    def retrieve_queue(self):
        queue = self.inlane_halting_vehicle_number

        return queue

    def retrieve_waiting_time(self):
        waiting_time = self.inlane_waiting_time

        return waiting_time
    
    def retrieve_left_time(self):
        if self.schedule[-1] == -1:
            return len(self.schedule)
        else:
            return len(self.schedule) - 1 + self.schedule[-1]


class SUMOEnv(gym.Env):

    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(self,
                 yellow,
                 num_stage: int,
                 num_agent: int,
                 use_gui: bool,
                 net_file: str,
                 route_file: str,
                 addition_file: str,
                 min_green: int = 10,
                 max_green: int = 40,
                 sumo_seed: Union[str, int] = "random",
                 max_depart_delay: int = -1,
                 waiting_time_memory: int = 1000,
                 time_to_teleport: int = -1,
                 max_step_round: int = 10000,
                 max_step_sample: int = 1000000,
                 observation_pattern: str = "queue",
                 ):
        super(SUMOEnv, self).__init__()

        self.yellow = yellow
        self.use_gui = use_gui
        self.net = net_file
        self.route = route_file
        self.addition = addition_file
        self.sumo_seed = sumo_seed
        self.num_stage = num_stage
        self.num_agent = num_agent
        self.min_green = min_green
        self.max_green = max_green
        self.max_step_episode = max_step_round
        self.max_step_sample = max_step_sample
        self.observation_pattern = observation_pattern

        if self.use_gui or self.render_mode is not None:
            self.sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self.sumo_binary = sumolib.checkBinary("sumo")
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.waiting_time_memory = waiting_time_memory  # Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime)
        self.time_to_teleport = time_to_teleport
        self.label = str(SUMOEnv.CONNECTION_LABEL)
        SUMOEnv.CONNECTION_LABEL += 1  # Increments itself when an instance is initialized

        self._step = 0
        self.queue_old = np.zeros((self.num_agent, ), dtype=np.int32)
        self.queue_cur = np.zeros((self.num_agent, ), dtype=np.int32)
        self.global_reward = np.zeros((self.num_agent, ), dtype=np.float32)
        self.sumo = None
        self.tl_ids = None
        self.tls = None
        self.terminated = False
        self.agents_to_update = None

    @property
    def observation_space(self):
        """Return the observation space of a traffic signal.
        Only used in case of single-agent environment.
        """
        return spaces.Box(low=-100, high=500, shape=(self.num_agent * self.num_stage,), dtype=np.int32)

    @property
    def action_space(self):
        """Return the action space of a traffic signal.
        Only used in case of single-agent environment.
        """
        return spaces.Tuple((spaces.MultiDiscrete(np.array([self.num_stage] * self.num_agent)),
                             spaces.Box(low=self.min_green, high=self.max_green, shape=(self.num_agent,),
                                        dtype=np.int64)))

    def step(self, action):
        """
        :param action:
        :return:
        """
        [self.tls[i].set_stage_duration(action[0][i], action[1][i]) for i in range(self.num_agent) if self.agents_to_update[i]]

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

    def start_simulation(self):
        """
        Start the sumo simulation according to the sumo commend.
        :return:
        """
        sumo_cmd = [
            self.sumo_binary,
            "-n",
            self.net,
            "-r",
            self.route,
            "-a",
            self.addition,
            "--max-depart-delay",
            str(self.max_depart_delay),
            "--waiting-time-memory",
            str(self.waiting_time_memory),
            "--time-to-teleport",
            str(self.time_to_teleport),
            "--no-warnings",
            str(True),
        ]

        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])

        if self.use_gui or self.render_mode is not None:
            sumo_cmd.extend(["--start", "--quit-on-end"])

        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)

        if self.use_gui or self.render_mode is not None:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

        self.tl_ids = list(self.sumo.trafficlight.getIDList())
        self.tls = [TrafficSignal(tl_id, yellow, self.sumo) for tl_id, yellow in zip(self.tl_ids, self.yellow)]

    def reset(self, seed: Optional[int] = None, **kwargs):
        """

        :param seed:
        :param kwargs:
        :return:
        """
        super(SUMOEnv, self).reset(seed=seed, **kwargs)
        if self._step != 0:
            self.close()
        self.agents_to_update = np.ones(self.num_agent, dtype=np.int32)
        self.terminated = False
        self._step = 0

        if seed is not None:
            self.sumo_seed = seed
        self.start_simulation()

        [tl.get_subscription_result() for tl in self.tls]

        if self.observation_pattern == 'queue':
            observation = np.array([tl.retrieve_queue() for tl in self.tls]).flatten()
        elif self.observation_pattern == 'pressure':
            observation = np.array([tl.retrieve_pressure() for tl in self.tls]).flatten()
        else:
            raise NotImplementedError

        info = {'agents_to_update': self.agents_to_update, 'terminated': self.terminated, 'queue': np.array([tl.retrieve_queue() for tl in self.tls]).sum(), 
                'left_time': np.array([0 for tl in self.tls]), 'global_reward': self.global_reward}
        return observation, info

    def close(self):
        """
        Close the environment and stop the SUMO simulation.
        :return:
        """

        if self.sumo is None:
            return

        if not LIBSUMO:
            traci.switch(self.label)
        traci.close()


if __name__ == "__main__":
    env = SUMOEnv(yellow=[3, 3, 3, 3, 3, 3, 3],
                  num_stage=8,
                  num_agent=7,
                  use_gui=True,
                  net_file='envs/Metro.net.xml',
                  route_file='envs/Metro.rou.xml',
                  addition_file='envs/Metro.add.xml',
                  observation_pattern='queue',
                  )
    env.reset()
    while True:
        action = env.action_space.sample()
        # action[0][0] = 0
        # action[0][1] = 0
        # action[0][2] = 0
        # action[0][3] = 0
        # action[0][4] = 0
        # action[0][5] = 0
        # action[0][6] = 0
        ts = TrafficSignal(env.tl_ids[0], 3, env.sumo)
        ts.get_subscription_result()
        obs_ = ts.retrieve_pressure()
        # print(obs_)
        # rew = ts.retrieve_reward()

        obs, rew, ter, trun, info = env.step(action)
        if ter:
            break
