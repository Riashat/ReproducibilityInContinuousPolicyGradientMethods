#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
import argparse

from mlp_policy import MlpPolicy
import pposgd_simple

parser = argparse.ArgumentParser()
parser.add_argument("--network_size", type=int, default=64)
parser.add_argument("--network_activation", type=str, default="tanh")
args = parser.parse_args()


def train(env_id, num_timesteps, seed):

    U.make_session(num_cpu=1).__enter__()
    logger.session().__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
            hid_size=args.network_size, num_hid_layers=2, activation=args.network_activation)


    env = bench.Monitor(env, osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn, 
            max_timesteps=num_timesteps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95,
        )
    env.close()


def main():
    train('Hopper-v1', num_timesteps=1e6, seed=0)


if __name__ == '__main__':
    main()
