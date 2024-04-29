import argparse
import gymnasium as gym
import numpy as np
import torch

from src.hppo.HPPO import *
from utils import *
from multiprocessing import Pool


class Trainer(object):
    """
    A RL trainer.
    """

    def __init__(self, args):

        self.device = args.device
        self.max_episodes = args.max_episodes
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.rolling_score_window = args.rolling_score_window
        self.agent_save_freq = args.agent_save_freq
        self.agent_update_freq = args.agent_update_freq
        self.action_space_pattern = args.action_space_pattern
        self.sumocfg = args.sumocfg

        # agent's hyperparameters
        self.mid_dim = args.mid_dim
        self.lr_actor = args.lr_actor
        self.lr_critic = args.lr_actor_param
        self.lr_std = args.lr_std
        self.lr_decay_rate = args.lr_decay_rate
        self.target_kl_dis = args.target_kl_dis
        self.target_kl_con = args.target_kl_con
        self.gamma = args.gamma
        self.lam = args.lam
        self.epochs_update = args.epochs_update
        self.v_iters = args.v_iters
        self.eps_clip = args.eps_clip
        self.max_norm_grad = args.max_norm_grad
        self.init_log_std = args.init_log_std
        self.coeff_dist_entropy = args.coeff_dist_entropy
        self.random_seed = args.random_seed
        self.if_use_active_selection = args.if_use_active_selection
        self.init_bonus = args.init_bonus

        # For save
        self.file_to_save = 'data/'
        if not os.path.exists(self.file_to_save):
            os.makedirs(self.file_to_save)
        self.record_mark = args.record_mark
        self.policy_save = os.path.join(self.file_to_save,
                                        'policy/{}/{}'.format(self.record_mark, self.action_space_pattern))
        self.results_save = os.path.join(self.file_to_save,
                                         'results/{}/{}'.format(self.record_mark, self.action_space_pattern))
        self.rolling_scores_save = os.path.join(self.file_to_save, 'rolling_scores/{}/{}'.format(self.record_mark,
                                                                                                 self.action_space_pattern))

        self.num_agent = args.num_agent
        self.num_stage = args.num_stage
        self.yellow = args.yellow
        self.delta_time = args.delta_time
        self.max_green = args.max_green
        self.min_green = args.min_green
        self.pattern = args.pattern
        os.makedirs(self.policy_save, exist_ok=True)
        os.makedirs(self.results_save, exist_ok=True)
        os.makedirs(self.rolling_scores_save, exist_ok=True)
        self.obs_dim = 8
        self.history = [{} for i in range(self.num_agent)]

        # The stage indicator for continuous action space
        # And for continuous the number of stages are set to be 4!!!.

        self.indicator = np.random.randint(self.num_stage, size=self.num_agent)

    
    def push_history_hyar(self, idx, obs, residual_obs, act, param_act):
        self.history[idx]['obs'] = obs
        self.history[idx]['residual_obs'] = residual_obs
        self.history[idx]['act'] = act
        self.history[idx]['param_act'] = param_act
        
    def push_history_hybrid(self, idx, obs, act_dis, act_con, logp_act_dis, logp_act_con, val):
        self.history[idx]['obs'] = obs
        self.history[idx]['act_dis'] = act_dis
        self.history[idx]['act_con'] = act_con
        self.history[idx]['logp_act_dis'] = logp_act_dis
        self.history[idx]['logp_act_con'] = logp_act_con
        self.history[idx]['val'] = val

    def push_history_continuous(self, idx, obs, act_con, logp_act_con, val):
        self.history[idx]['obs'] = obs
        self.history[idx]['act_con'] = act_con
        self.history[idx]['logp_act_con'] = logp_act_con
        self.history[idx]['val'] = val

    def mapping(self, head):
        return (np.tanh(head) + 1) / 2 * (self.max_green - self.min_green) + self.min_green

    def unbatchify(self, value_action_logp: dict, agents_to_update: np.ndarray, state):

        if self.action_space_pattern == 'hybrid':
            values = np.array([value_action_logp[i][0] if agents_to_update[i] == 1 else 0 for i in range(self.num_agent)])
            stages = np.array([value_action_logp[i][1][0] if agents_to_update[i] == 1 else -1 for i in range(self.num_agent)])
            durations = np.array([self.mapping(value_action_logp[i][1][1]) if agents_to_update[i] == 1 else 0 for i in range(self.num_agent)], dtype=np.int64)
            logp_stages = np.array([value_action_logp[i][2][0] if agents_to_update[i] == 1 else 0 for i in range(self.num_agent)])
            logp_durations = np.array([value_action_logp[i][2][1] if agents_to_update[i] == 1 else 0 for i in range(self.num_agent)])
            actions = np.array([stages, durations])
            logp_actions = np.array([logp_stages, logp_durations])

        elif self.action_space_pattern == 'continuous':

            stages = self.indicator
            values = np.array([value_action_logp[i][0][stages[i]] if agents_to_update[i] == 1 else 0 for i in range(self.num_agent)])
            durations = np.array([self.mapping(value_action_logp[i][1][stages[i]]) if agents_to_update[i] == 1 else 0 for i in range(self.num_agent)], dtype=np.int64)
            actions = np.array([stages, durations])
            logp_actions = np.array([value_action_logp[i][2][stages[i]] if agents_to_update[i] == 1 else 0 for i in range(self.num_agent)])

        elif self.action_space_pattern == 'discrete':
            values = np.array([value_action_logp[i][0] for i in range(self.num_agent)])
            stages = np.array([value_action_logp[i][1] for i in range(self.num_agent)])
            durations = np.array([self.delta_time] * self.num_agent)
            actions = np.array([stages, durations])
            logp_actions = np.array([value_action_logp[i][2] for i in range(self.num_agent)])
        
        else:
            pass
        return values, actions, logp_actions

    def visualize_one_episode(self, i_episode, worker_idx, episode_score):
        queue_pic_save = os.path.join(self.results_save, 'queue_episode_{}_{}.jpg'.format(i_episode, worker_idx))
        delay_pic_save = os.path.join(self.results_save, 'delay_episode_{}_{}.jpg'.format(i_episode, worker_idx))
        visualize_results_per_run(episode_score['queue'], self.action_space_pattern, queue_pic_save, 'Avg Queue')
        visualize_results_per_run(episode_score['delay'], self.action_space_pattern, delay_pic_save, 'Avg Delay')

    def save_data(self, rolling_score, normalization_params, worker_idx):
        path_to_save_npz = os.path.join(self.file_to_save,
                                        'rolling_data/{}/{}/'.format(self.record_mark, self.action_space_pattern))
        os.makedirs(path_to_save_npz, exist_ok=True)
        np.savez(path_to_save_npz + 'queue_{}'.format(worker_idx), queue=rolling_score['queue'])
        np.savez(path_to_save_npz + 'delay_{}'.format(worker_idx), delay=rolling_score['delay'])
        np.savez(path_to_save_npz + 'mean_{}'.format(worker_idx), mean=normalization_params['mean'])
        np.savez(path_to_save_npz + 'std_{}'.format(worker_idx), std=normalization_params['std'])

    def plot(self):
        delay, queue = extract_over_all_rs(self.record_mark, [21, 22, 23, 24, 26])
        visualize_overall_agent_results(delay, ['continuous'],
                                        'data/rolling_scores/{}/delay.jpg'.format(self.record_mark),
                                        'rolling score of delay', 'avg delay')
        visualize_overall_agent_results(queue, ['continuous'],
                                        'data/rolling_scores/{}/queue.jpg'.format(self.record_mark),
                                        'rolling score of queue', 'avg queue')

    def initialize_agents(self, random_seed):
        """
        Initialize environment and agent.

        :param random_seed: could be regarded as worker index
        :return: instance of agent and env
        """
        agents = None
        if self.action_space_pattern == 'discrete':
            agents = [PPO_Discrete(self.obs_dim, self.num_stage, self.mid_dim, self.lr_actor, self.lr_critic,
                                   self.lr_decay_rate, self.buffer_size, self.target_kl_dis, self.target_kl_con, self.gamma, self.lam, self.epochs_update,
                                   self.v_iters, self.eps_clip, self.max_norm_grad, self.coeff_dist_entropy, random_seed, self.device)
                      for i in range(self.num_agent)]

        elif self.action_space_pattern == 'continuous':
            agents = [PPO_Continuous(self.obs_dim, self.num_stage, self.mid_dim, self.lr_actor, self.lr_critic, self.lr_decay_rate,
                                     self.buffer_size, self.target_kl_dis, self.target_kl_con, self.gamma, self.lam, self.epochs_update, self.v_iters,
                                     self.eps_clip, self.max_norm_grad, self.coeff_dist_entropy, random_seed, self.device,
                                     self.lr_std, self.init_log_std)
                      for i in range(self.num_agent)]

        elif self.action_space_pattern == 'hybrid':
            agents = [PPO_Hybrid(self.obs_dim, self.num_stage, self.mid_dim, self.lr_actor, self.lr_critic, self.lr_decay_rate,
                                 self.buffer_size, self.target_kl_dis, self.target_kl_con, self.gamma, self.lam, self.epochs_update,self.v_iters,
                                 self.eps_clip, self.max_norm_grad, self.coeff_dist_entropy, random_seed, self.device,
                                 self.lr_std, self.init_log_std, self.if_use_active_selection, self.init_bonus)
                      for i in range(self.num_agent)]
            
        return agents

    def train(self, worker_idx):
        """

        :param worker_idx:
        :return:
        """
        env = gym.make('sumo-rl-v1',
                       yellow=[self.yellow] * self.num_agent,
                       num_agent=self.num_agent,
                       num_stage=self.num_stage,
                       use_gui=False,
                       net_file='envs/4_4.net.xml',
                       route_file='envs/4_4_high.rou.xml',
                       addition_file='envs/4_4.add.xml',
                       max_step_round=3600,
                       observation_pattern=self.pattern,
                       )

        agents = self.initialize_agents(worker_idx)
        monitor = Monitor(self.rolling_score_window)

        norm_mean = np.zeros(shape=(self.num_agent, self.obs_dim))
        norm_std = np.ones(shape=(self.num_agent, self.obs_dim))

        i_episode = 0

        ### TRAINING LOGIC ###
        while i_episode < self.max_episodes:
            # collect an episode
            with torch.no_grad():
                state, info = env.reset()
                next_state = state
                agents_to_update = info['agents_to_update']

                while True:
                    # Every update, we will normalize the state_norm(the input of the actor_con and critic) by
                    # mean and std retrieve from the last update's buf, in other word observations normalization
                    observations = state.reshape(self.num_agent, -1)
                    observations_norm = (observations - norm_mean) / np.maximum(norm_std, 1e-6)
                    # Select action with policy
                    value_action_logp = {i: agents[i].select_action(observations_norm[i]) for i in range(self.num_agent) if info['agents_to_update'][i] == 1}
                    values, actions, logp_actions = self.unbatchify(value_action_logp, info['agents_to_update'], observations_norm)

                    next_state, reward, done, truncated, info = env.step(actions)

                    if self.action_space_pattern == 'continuous':
                        [self.push_history_continuous(i, observations[i], value_action_logp[i][1], logp_actions[i], values[i])
                         for i in range(self.num_agent) if agents_to_update[i] == 1]
                        [agents[i].buffer.store_con(self.history[i]['obs'], self.history[i]['act_con'], reward[i], self.history[i]['val'], self.history[i]['logp_act_con'], self.indicator[i])
                         for i in range(self.num_agent) if info['agents_to_update'][i] == 1]
                        self.indicator = (self.indicator + info['agents_to_update']) % self.num_stage

                    elif self.action_space_pattern == 'discrete':
                        [agents[i].buffer.store_dis(observations[i], actions[0][i], reward[i], values[i], logp_actions[i])
                         for i in range(self.num_agent)]

                    elif self.action_space_pattern == 'hybrid':
                        [self.push_history_hybrid(i, observations[i], actions[0][i], value_action_logp[i][1][1], logp_actions[0][i], logp_actions[1][i], values[i])
                         for i in range(self.num_agent) if agents_to_update[i] == 1]

                        [agents[i].buffer.store_hybrid(self.history[i]['obs'], self.history[i]['act_dis'], self.history[i]['act_con'], reward[i], self.history[i]['val'], self.history[i]['logp_act_dis'], self.history[i]['logp_act_con'])
                         for i in range(self.num_agent) if info['agents_to_update'][i] == 1]
                        
                        

                    # In continuous control pattern, it's meaningful to store "ptr" into the buffer.
                    # while in discrete and hybrid, it's not.
                    # But for consistency, we will include this process in all control patterns.

                    # update observation
                    state = next_state
                    agents_to_update = info['agents_to_update']

                    # for evaluation
                    # monitor.push_into_monitor(sum(info['queue']), sum(info['waiting_time']))


                    monitor.push_into_monitor(info['queue'], info['queue'])

                    if info['terminated']:
                        i_episode += 1
                        [agents[i].buffer.finish_path(0) for i in range(self.num_agent)]
                        break

            if i_episode % self.agent_update_freq == 0:
                # For the trick of observation normalization
                for i in range(self.num_agent):
                    tmp = agents[i].buffer.filter()[0]
                    norm_mean[i] = np.tile(agents[i].buffer.filter()[0], self.obs_dim)
                    norm_std[i] = np.tile(agents[i].buffer.filter()[1], self.obs_dim)
                if i_episode > self.agent_save_freq:
                    [agents[i].update(self.batch_size) for i in range(self.num_agent)]
                [agents[i].buffer.clear() for i in range(self.num_agent)]

            if i_episode % self.agent_save_freq == 0:
                file_to_save_policy = os.path.join(self.policy_save, 'i_episode{}_{}'.format(i_episode, worker_idx))
                print('-----------------------------------------------------------------------------------')
                print('saving model at:', file_to_save_policy)
                [agents[i].save(file_to_save_policy + '_{}'.format(i)) for i in range(self.num_agent)]
                print('model saved')
                print('-----------------------------------------------------------------------------------')

            # record episodes score and rolling score
            episode_score, rolling_score = monitor.output_from_monitor()

            if i_episode % 5 == 0:
                self.save_data(rolling_score, {'mean': norm_mean, 'std': norm_std}, worker_idx)
                self.visualize_one_episode(i_episode, worker_idx, episode_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Device.')
    parser.add_argument('--max_episodes', type=int, default=500, help='The max episodes per agent per run.')
    parser.add_argument('--buffer_size', type=int, default=20000, help='The maximum size of the PPOBuffer.')
    parser.add_argument('--batch_size', type=int, default=512, help='The sample batch size.')
    parser.add_argument('--rolling_score_window', type=int, default=5,
                        help='Mean of last rolling_score_window.')  # TODO is there any need?
    parser.add_argument('--agent_save_freq', type=int, default=5, help='The frequency of the agent saving.')
    parser.add_argument('--agent_update_freq', type=int, default=5, help='The frequency of the agent updating.')
    parser.add_argument('--lr_actor', type=float, default=0.0003, help='The learning rate of actor_con.')   # carefully!
    parser.add_argument('--lr_actor_param', type=float, default=0.001, help='The learning rate of critic.')
    parser.add_argument('--lr_std', type=float, default=0.004, help='The learning rate of log_std.')
    parser.add_argument('--lr_decay_rate', type=float, default=0.995, help='Factor of learning rate decay.')
    parser.add_argument('--mid_dim', type=list, default=[256, 128, 64], help='The middle dimensions of both nets.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted of future rewards.')
    parser.add_argument('--lam', type=float, default=0.8,
                        help='Lambda for GAE-Lambda. (Always between 0 and 1, close to 1.)')
    parser.add_argument('--epochs_update', type=int, default=20,
                        help='Maximum number of gradient descent steps to take on policy loss per epoch. (Early stopping may cause optimizer to take fewer than this.)')
    parser.add_argument('--v_iters', type=int, default=1,
                        help='Number of gradient descent steps to take on value function per epoch.')
    parser.add_argument('--target_kl_dis', type=float, default=0.025,
                        help='Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping. (Usually small, 0.01 or 0.05.)')
    parser.add_argument('--target_kl_con', type=float, default=0.05,
                        help='Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping. (Usually small, 0.01 or 0.05.)')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='The clip ratio when calculate surr.')
    parser.add_argument('--max_norm_grad', type=float, default=5.0, help='max norm of the gradients.')
    parser.add_argument('--init_log_std', type=float, default=-1.0,
                        help='The initial log_std of Normal in continuous pattern.')
    parser.add_argument('--coeff_dist_entropy', type=float, default=0.005,
                        help='The coefficient of distribution entropy.')
    parser.add_argument('--random_seed', type=int, default=1, help='The random seed.')
    parser.add_argument('--action_space_pattern', type=str, default='hybrid',
                        help='The control pattern of the action.')
    parser.add_argument('--record_mark', type=str, default='renaissance',
                        help='The mark that differentiates different experiments.')
    parser.add_argument('--if_use_active_selection', type=bool, default=False,
                        help='Whether use active selection in the exploration.')
    parser.add_argument('--init_bonus', type=float, default=0.01, help='The initial active selection bonus.')
    parser.add_argument('--sumocfg', type=str, default='Env4', help='The initial active selection bonus.')
    parser.add_argument('--num_stage', type=int, default=8)
    parser.add_argument('--num_agent', type=int, default=16)
    parser.add_argument('--yellow', type=int, default=3)
    parser.add_argument('--delta_time', type=int, default=15)
    parser.add_argument('--max_green', type=int, default=40)
    parser.add_argument('--min_green', type=int, default=10)
    parser.add_argument('--pattern', type=str, default='queue')

    
    args = parser.parse_args()

    # args log
    argsDict = args.__dict__
    with open('args_log/{}_{}.txt'.format(args.record_mark, args.action_space_pattern), 'w') as f:
        f.writelines('------------ start ------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ':' + str(value) + '\n')
        f.writelines('------------ end ------------')

    # training through multiprocess
    trainer = Trainer(args)
    trainer.train(0)

    # trainer.plot()

    # args_tuple = [[31], [32], [33], [34], [35], [36]]
    # pool = Pool(processes=6)
    # for arg in args_tuple:
    #     pool.apply_async(trainer.train, arg)
    # pool.close()
    # pool.join()
