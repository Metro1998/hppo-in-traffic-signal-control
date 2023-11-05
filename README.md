
TODO LIST (if you have any other requests or feedback, please let me know by opening an issue.)

- [ ] **Implement parallel acceleration based on EnvPool**

- [ ] **Develop a state representation based on VectorNet**

- [ ] **Create a new agent based on PPO+HyAR**






# a-simple-implementation-of-hppo

In this repository, utilizing hybrid proximal policy optimization ([H-PPO](https://dl.acm.org/doi/10.5555/3367243.3367356)), we have implemented the synchronous optimization of the signal staging (discrete parameter) and its corresponding duration (continuous parameter).  <code>PPO-family.py</code> and <code>environments</code> are involved in this repository.  

Rollout buffer, network architectures, and PPO classes are defined in the <code>PPO_family.py</code>. In the specific implementation, we have referred to [OpenAI](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo) and [ikostrikov](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py). We have also considered the tricks in [Engstrom et al., 2020](https://arxiv.org/abs/2005.12729) and [Andrychowicz et al., 2020](https://arxiv.org/abs/2006.05990v1).

In the <code>environment</code> folder, we define the traffic demand as well as the simulation configuration, for example, <code>Env1_rou.xml</code> and <code>Env1_sumocfg</code>, to make the sumo work. Four environment classes are also included, which have been encapsulated to provide interactive envs. Of course, if you are not interested in traffic signal control, you can ignore these envs and write on your own.

Finally, we present the overview of H-PPO’s architecture as well as the partial training results.

![](https://github.com/Metro1998/a-simple-implementation-of-hppo/blob/main/pictures/overview.png)

Average queue length of Env #9             |  Average delay of Env #9
:-------------------------:|:-------------------------:
![](https://github.com/Metro1998/a-simple-implementation-of-hppo/blob/main/pictures/queue_9.png)  |  ![](https://github.com/Metro1998/a-simple-implementation-of-hppo/blob/main/pictures/delay_9.png)

