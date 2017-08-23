#!/usr/bin/env python
import argparse
import logging
import os
import tensorflow as tf
import gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from acktr_cont import learn
from policies import GaussianMlpPolicy
from value_functions import NeuralNetValueFunction
import argparse


def train(env_id, network_size, network_activation, num_timesteps, seed):
    env=gym.make(env_id)
    if logger.get_dir():
        env = bench.Monitor(env, os.path.join(logger.get_dir(), "monitor.json"))
    set_global_seeds(seed)
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)

    with tf.Session(config=tf.ConfigProto()) as session:
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]

        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim, network_size, network_activation)

        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim, network_size, network_activation)

        learn(env, policy=policy, vf=vf,
            gamma=0.99, lam=0.97, timesteps_per_batch=2500,
            desired_kl=0.002,
            num_timesteps=num_timesteps, animate=False)

        env.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Mujoco benchmark.')
    parser.add_argument('--env_id', type=str, default="Hopper-v1")
    parser.add_argument("--network_size", type=int, default=64)
    parser.add_argument("--network_activation", type=str, default="tanh")
    args = parser.parse_args()
    train(args.env_id, args.network_size, args.network_activation, num_timesteps=1e6, seed=1)
