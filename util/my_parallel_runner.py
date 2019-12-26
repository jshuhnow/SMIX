from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np

class MyParallelRunner:

    def __init__(self, args, logger, env, num_parallel=8):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        # assert self.batch_size == 1

        # self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.env = env
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        self.num_parallel = num_parallel

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

        self.terminated = False
        self.episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

    def get_actions(self):
        pre_transition_data = {
            "state": [self.env.get_state()] * self.num_parallel ,
            "avail_actions": [self.env.get_avail_actions()] * self.num_parallel ,
            "obs": [self.env.get_obs()] * self.num_parallel
        }
        self.batch.update(pre_transition_data, ts=self.t)

        # Pass the entire batch of experiences up till now to the agents
        # Receive the actions for each agent at this timestep in a batch of size 1
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=True)
        return actions

    def post_action(self, reward, terminated, env_info, actions):
        self.episode_return += reward

        post_transition_data = {
            "actions": actions,
            "reward": [(reward,) * self.num_parallel ],
            "terminated": [(terminated != env_info.get("episode_limit", False),) * self.num_parallel ],
        }

        self.batch.update(post_transition_data, ts=self.t)
        self.t += 1

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
