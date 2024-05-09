import gymnasium as gym
import numpy as np
import libtraci as traci
import traci
import sumolib
import os
import time

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
    """

    def __init__(self, tl_id, pattern, yellow, sumo):

        self.id = tl_id
        self.pattern = pattern
        self.yellow = yellow
        self.sumo = sumo

        # The schedule is responsible for the automatic timing for the incoming green stage.
        # | 0 | 0 | 0 | 16 |
        # | yellow len| when 16 is dequeued, the stage is automatically transferred to the green stage and 16 is for duration.
        self.schedule = deque()
        self.duration = None

        # Links is relative with connections defined in the rou.xml, what's more the connection definition should be
        # relative with traffic state definition. Therefore, there is no restriction that the connection should start
        # at north and step clockwise then.
        all_lanes = self.sumo.trafficlight.getControlledLinks(self.id)
        self.in_lanes = [conn[0][0] for conn in all_lanes]

        # Delete the right turn movement.

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
        self.outlane_halting_vehicle_number = None
        self.stage_old = np.random.randint(0, 8)

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
        if self.stage_old != stage:
            yellow_stage = int(self.mapping[self.stage_old][stage])
            self.sumo.trafficlight.setPhase(self.id, yellow_stage)
            for i in range(self.yellow):
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
            [list(self.sumo.lane.getSubscriptionResults(lane_id).values())[0] for lane_id in self.in_lanes],
            dtype=np.int32)
        self.outlane_halting_vehicle_number = np.array(
            [list(self.sumo.lane.getSubscriptionResults(lane_id).values())[0] for lane_id in self.out_lanes],
            dtype=np.int32)

    def retrieve_reward(self):
        if not isinstance(self.inlane_halting_vehicle_number_old, np.ndarray):
            reward = -sum(self.inlane_halting_vehicle_number)
        else:
            reward = sum(self.inlane_halting_vehicle_number_old) - sum(self.inlane_halting_vehicle_number)

        self.inlane_halting_vehicle_number_old = self.inlane_halting_vehicle_number

        return reward

    def retrieve_info(self):
        queue = self.inlane_halting_vehicle_number

        return queue

    def retrieve_queue(self):
        if self.pattern == 'pressure':
            observation = self.inlane_halting_vehicle_number - self.outlane_halting_vehicle_number
        elif self.pattern == 'queue':
            observation = self.inlane_halting_vehicle_number

        return observation

    def retrieve_stage(self):

        return self.stage_old
    
    def retrieve_left_time(self):
        return len(self.schedule)

    def clear_schedule(self):
        self.schedule.clear()


class SUMOEnv(gym.Env):
    CONNECTION_LABEL = 0

    def __init__(self,
                 yellow,
                 num_agent: int,
                 use_gui: bool,
                 net_file: str,
                 route_file: str,
                 addition_file: str,
                 min_green: int = 10,
                 max_green: int = 40,
                 pattern: str = 'queue',
                 sumo_seed: Union[str, int] = "random",
                 max_episode_step: int = 10000,
                 max_sample_step: int = 1000,
                 comment: str = "test",
                 ):

        self.yellow = yellow
        self.num_agent = num_agent
        self.use_gui = use_gui
        self.net_file = net_file
        self.route_file = route_file
        self.addition_file = addition_file
        self.sumo_seed = sumo_seed
        self.comment = comment

        self.num_stage = 8
        self.min_green = min_green
        self.max_green = max_green
        self.pattern = pattern
        self.max_episode_step = max_episode_step
        self.max_sample_step = max_sample_step

        if self.use_gui or self.render_mode is not None:
            self.sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self.sumo_binary = sumolib.checkBinary("sumo")
        self.label = str(SUMOEnv.CONNECTION_LABEL)
        SUMOEnv.CONNECTION_LABEL += 1  # Increments itself when an instance is initialized

        self.episode_step = 0
        self.sample_step = 0
        self.episode = 0
        self.sumo = None
        self.tl_ids = None
        self.tls = None
        self.rewards = None
        self.terminated = False
        self.trunc = False
        self.critical_step_idx = None
        self.agents_to_update = np.ones(self.num_agent, dtype=np.int32)

    @property
    def observation_space(self):
        """Return the observation space of a traffic signal.
        Only used in case of single-agent environment.
        """
        return spaces.Dict({
            'stage': spaces.MultiDiscrete(np.array([self.num_stage] * self.num_agent), dtype=np.int32),
            'queue': spaces.Box(low=-200, high=200, shape=(self.num_agent * self.num_stage, ), dtype=np.int32),
        })

    @property
    def action_space(self):
        """Return the action space of a traffic signal.
        Only used in case of single-agent environment.
        """
        return spaces.Dict({
            'stage': spaces.MultiDiscrete(np.array([self.num_stage] * self.num_agent), dtype=np.int32),
            'duration': spaces.Box(low=self.min_green, high=self.max_green, shape=(self.num_agent,), dtype=np.int32)
        })

    def step(self, action):
        """

        :param action:
        :return:
        """
        action = np.stack((action['stage'], action['duration']), axis=1)
        action_executed = action[self.agents_to_update == 1]
        tls_executed = [tl for tl, a in zip(self.tls, self.agents_to_update) if a == 1]
        for a, tl in zip(action_executed, tls_executed):
            tl.set_stage_duration(a[0], a[1])

        if self.sample_step == 0: self.critical_step_idx = [[] for _ in range(self.num_agent)]
        
        while True:
            self.sumo.simulationStep()

            # Automatically execute the transition from the yellow stage to green stage, and simultaneously set the end indicator -1.
            # Moreover, check() will return the front of the schedule.
            checks = [tl.check() for tl in self.tls]

            # Pop the most left element of the schedule.
            [tl.pop() for tl in self.tls]
            
            self.episode_step += 1
            self.terminated = (self.episode_step >= self.max_episode_step)
            
            # ids are agents who should act right now.
            if -1 in checks or self.terminated: 
                self.agents_to_update = -np.array(checks, dtype=np.int64)
                
                left_time = np.array([tl.retrieve_left_time() for tl in self.tls])
                [tl.get_subscription_result() for tl in self.tls]
                reward = np.array([tl.retrieve_reward() for tl in self.tls])
                # critical_step_idx (list) is the index for recomputing reward.
                # For example, if the reward fraction for one agent is [3, 2, 1, 5, 6], and the critical_step_idx is [3, 5], then the final reward for this agent is [6, 3, 1, 11, 6]
                [self.critical_step_idx[i].append(self.sample_step) for i in range(self.num_agent) if self.agents_to_update[i] == 1 and self.sample_step > 0]

                self.sample_step += 1
                if (self.sample_step >= self.max_sample_step):
                    self.sample_step = 0
                break

        observation = { 'queue': np.array([tl.retrieve_queue() for tl in self.tls]).flatten(),
                        'stage': np.array([tl.retrieve_stage() for tl in self.tls]).flatten()}
        
        info = {}
        # For performance evaluation
        info['queue'] = np.array([sum(tl.retrieve_info()) for tl in self.tls])

        # For reward calculation
        info['critical_step_idx'] = self.critical_step_idx

        # For policy update
        info['agents_to_update'] = self.agents_to_update
        
        # For left time
        info['left_time'] = left_time
        
        # For agents to update
        info['agents_to_update'] = self.agents_to_update
        
        info['trunc'] = self.sample_step == 0
        
        info['termi'] = self.terminated

        return observation, reward, False, False, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super(SUMOEnv, self).reset(seed=seed)

        if self.episode_step != 0:
            self.close()
        if seed is not None:
            self.sumo_seed = seed
        self.episode_step = 0
        self.sample_step = 0
        self.terminated = False
        self.critical_step_idx = [[] for _ in range(self.num_agent)]
        self.agents_to_update = np.ones(self.num_agent, dtype=np.int32)
        self.start_simulation()

        [tl.get_subscription_result() for tl in self.tls]
        [tl.clear_schedule() for tl in self.tls]
        observation = {'queue': np.array([tl.retrieve_queue() for tl in self.tls]).flatten(),
                       'stage': np.array([tl.retrieve_stage() for tl in self.tls]).flatten()}
        
        info = {}

        # For agents to update
        info['agents_to_update'] = self.agents_to_update

        info['trunc'] = False

        info['left_time'] = np.array([tl.retrieve_left_time() for tl in self.tls])
        
        
        return observation, info

    def start_simulation(self):
        """
        Start the sumo simulation according to the sumo commend.
        :return:
        """
        sumo_cmd = [
            self.sumo_binary,
            "-n",
            self.net_file,
            "-r",
            self.route_file,
            "-a",
            self.addition_file,
            # "--no-warnings",
            # "true",
            # default settings
            "--max-depart-delay",
            "-1",
            "--waiting-time-memory",
            "1000",
            "--time-to-teleport",
            "-1",
            "--end",
            "7200",
            # "--device.tripinfo.probability",
            # "1",
            # "--mesosim",
            # str(True),
            # "--tripinfo-output",
            # "runs/tripinfo/" + self.comment + ".xml",
            # "--step-length",
            # "1",
            # "--default.action-step-length",
            # "1"
            # "--step-method.ballistic",
            # "True",
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
        self.tls = [TrafficSignal(tl_id, self.pattern, yellow, self.sumo) for tl_id, yellow in
                    zip(self.tl_ids, self.yellow)]

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
                  num_agent=7,
                  use_gui=True,
                  net_file='envs/Metro.net.xml',
                  route_file='envs/Metro.rou.xml',
                  addition_file='envs/Metro.add.xml')
    env.reset()
    while True:
        action = env.action_space.sample()
        _, _, _, _, _ = env.step(action)
