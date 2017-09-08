#!/usr/bin/env python
import argparse
import logging
import os
import tensorflow as tf
import gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.acktr.acktr_cont import learn
from policies import GaussianMlpPolicy
from value_functions import NeuralNetValueFunction

def train(env_id, policy_activation, value_activation, policy_size, value_size, num_timesteps, seed):
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
            vf = NeuralNetValueFunction(ob_dim, ac_dim, value_activation, value_size)
        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim, policy_activation, policy_size)

        learn(env, policy=policy, vf=vf,
            gamma=0.99, lam=0.97, timesteps_per_batch=2500,
            desired_kl=0.002,
            num_timesteps=num_timesteps, animate=False)

        env.close()

def main():
    parser = argparse.ArgumentParser(description='Run Mujoco benchmark.')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--env', help='environment ID', type=str, default="HalfCheetah-v1")
    parser.add_argument("--log_dir", type=str, default="./Results/")
    parser.add_argument("--activation_policy", type=str, default="relu")
    parser.add_argument("--activation_vf", type=str, default="relu")
    parser.add_argument("--policy_size", default=64, type=int)
    parser.add_argument("--vf_size", default=64, type=int)

    args = parser.parse_args()
    logger.configure(dir=args.log_dir)
    
    train(args.env, args.activation_policy, args.activation_vf, args.policy_size, args.vf_size, num_timesteps=2e6, seed=args.seed)

if __name__ == '__main__':
    main()
