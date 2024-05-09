
# Reinforcement learning for traffic signal control in hybrid action space

## H-PPO
In this repository, utilizing Hybrid Proximal Policy Optimization[^1] (H-PPO), we have implemented the synchronous optimization of the signal staging (discrete action) and its corresponding duration (continuous parameter). 

The rollout buffer, network architectures, and (H)PPO classes are defined in <code>src\hppo\HPPO.py</code>. In the specific implementation, we have referred to [repo:OpenAI](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo) and [repo:ikostrikov](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py). We have also considered the tricks in revisit papers[^2][^3][^4].

In the <code>envs\ </code>, we define the traffic demand as well as the simulation configuration, for example, <code>4_4.rou.xml</code> and <code>4_4.add.xml</code>, to make the sumo work. The environment class <code>SUMOEnv</code> is also included, which have been encapsulated to provide interactive envs (Please make sure you have registered it locally). Of course, if you are not interested in traffic signal control, you can ignore this env and write on your own. High performance and vectorized env, is under developed.

Finally, we present the overview of H-PPO’s architecture as well as the partial training results. And you can cite our paper[^5] as well.

![](https://github.com/Metro1998/a-simple-implementation-of-hppo/blob/main/pictures/overview.png)

Average queue length of Env #9             |  Average delay of Env #9
:-------------------------:|:-------------------------:
![](https://github.com/Metro1998/a-simple-implementation-of-hppo/blob/main/pictures/queue_9.png)  |  ![](https://github.com/Metro1998/a-simple-implementation-of-hppo/blob/main/pictures/delay_9.png)

## HyAR(TODO)

[^1]: [FAN Z, SU R, ZHANG W, et al. Hybrid actor-critic reinforcement learning in parameterized action
space[C]//Proceedings of the 28th International Joint Conference on Artificial Intelligence. 2019:
2279-2285.](https://dl.acm.org/doi/abs/10.5555/3367243.3367356)
[^2]: [HUANG S, DOSSA R F J, RAFFIN A, et al. The 37 implementation details of proximal policy
optimization[J]. The ICLR Blog Track, 2022.](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
[^3]: [ENGSTROM L, ILYAS A, SANTURKAR S, et al. Implementation Matters in Deep RL: A Case
Study on PPO and TRPO[C]//International Conference on Learning Representations. 2020.](https://openreview.net/forum?id=r1etN1rtPB)
[^4]: [ANDRYCHOWICZ M, RAICHUK A, STAŃCZYK P, et al. What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study[C]//ICLR 2021-Ninth International Conference on
Learning Representations. 2021.](https://arxiv.longhoe.net/abs/2006.05990)
[^5]: [LUO H, BIE Y, JIN S. Reinforcement Learning for Traffic Signal Control in Hybrid Action Space
[J]. IEEE Transactions on Intelligent Transportation Systems, 2024: 1-17.](https://ieeexplore.ieee.org/document/10379485)

