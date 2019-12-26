# reproduced from 'https://github.com/oxwhirl/smac/blob/master/smac/examples/random_agents.py'

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import yaml
import os

from smix.src.components.episode_buffer import ReplayBuffer
from smix.src.components.transforms import OneHot
from smix.src.controllers.basic_controller import BasicMAC

import torch as th
from types import SimpleNamespace as SN
from functools import partial
from smix.src.components.episode_buffer import EpisodeBatch
from smix.src.learners import REGISTRY as le_REGISTRY

DEFAULT_YAML_PATH = "smix/src/config/default.yaml"
ENVS_YAML_PATH = "smix/src/config/envs/sc2.yaml"
TIMESTEP_TO_LOAD = 0

class MyModel:
    def __init__(self,
                 model_path,
                 algs_yaml_path,
                 num_parallel
        ):
        self.model_path = model_path
        self.algs_yaml_path = algs_yaml_path
        self.num_parallel = num_parallel

    def _main(self):
        import collections
        from os.path import dirname, abspath
        from copy import deepcopy
        from utils.logging import get_logger
        from utils.logging import Logger

        def _get_config(path):
            with open(path, "r") as f:
                try:
                    config_dict = yaml.load(f)
                except yaml.YAMLError as exc:
                    assert False, "yaml error"
            return config_dict

        def recursive_dict_update(d, u):
            for k, v in u.items():
                if isinstance(v, collections.Mapping):
                    d[k] = recursive_dict_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        def config_copy(config):
            if isinstance(config, dict):
                return {k: config_copy(v) for k, v in config.items()}
            elif isinstance(config, list):
                return [config_copy(v) for v in config]
            else:
                return deepcopy(config)

        with open(DEFAULT_YAML_PATH) as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "default.yaml error"
        env_config = _get_config(ENVS_YAML_PATH)
        alg_config = _get_config(self.algs_yaml_path)

        config_dict = recursive_dict_update(config_dict, env_config)
        config_dict = recursive_dict_update(config_dict, alg_config)

        self.args = SN(**config_dict)

        from utils.logging import get_logger
        self.logger = get_logger()

        # TODO - SEEDING

    def _run(self):
        self.args.device = "cpu"

    def _run_sequential(self, env):

        from .my_eposide_runner import MyEpisodeRunner
        from .my_parallel_runner import MyParallelRunner

        if self.num_parallel > 1:
            runner = MyParallelRunner(self.args, self.logger, env, self.num_parallel)
        else:
            runner = MyEpisodeRunner()

        # Set up schemes and groups here
        env_info = runner.get_env_info()
        self.args.n_agents = env_info["n_agents"]
        self.args.n_actions = env_info["n_actions"]
        self.args.state_shape = env_info["state_shape"]

        # Default/Base scheme
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }

        groups = {
            "agents": self.args.n_agents
        }
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=self.args.n_actions)])
        }

        buffer = ReplayBuffer(scheme, groups, self.args.buffer_size, env_info["episode_limit"] + 1,
                              preprocess=preprocess,
                              device="cpu")

        # Setup multiagent controller here
        mac = BasicMAC(buffer.scheme, groups, self.args)

        # Give runner the scheme
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

        timesteps = []
        learner = le_REGISTRY[self.args.learner](mac, buffer.scheme, self.logger, self.args)

        learner.load_models(self.model_path)
        runner.t_env = TIMESTEP_TO_LOAD

        self.runner=runner

    def my_init(self, env):
        self._main()
        self._run()
        self._run_sequential(env)

        self.runner.reset()

    def _evaluate_sequential(self):
        pass
        # self.runner.run()

    def get_actions(self):
        return self.runner.get_actions()

    def post_action(self, reward,terminated,info, actions):
        return self.runner.post_action(reward,terminated,info, actions)






