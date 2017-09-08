import argparse
import time
import os
import logging
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
# import baselines.ddpg.training as training
import training
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *
from baselines.common import tf_util as U
import gym
import tensorflow as tf
from mpi4py import MPI



def run(env_id, seed, noise_type, layer_norm, evaluation, save_dir, num_experiments, nb_epochs, **kwargs):

    all_exp_eval_return = np.zeros(shape=(nb_epochs, num_experiments))

    all_seeds = [0, 101, 202, 303, 404]

    for e in range(num_experiments):

        print ("Experiment Number", e)

        # Configure things.
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0: logger.set_level(logger.DISABLED)

        # Create envs.
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), "%i.monitor.json"%rank))
        gym.logger.setLevel(logging.WARN)

        if evaluation and rank==0:
            eval_env = gym.make(env_id)
            eval_env = bench.Monitor(eval_env, logger.get_dir() and os.path.join(logger.get_dir(), "%i.monitor.json"%rank))
        else:
            eval_env = None

        # Parse noise_type
        action_noise = None
        param_noise = None
        nb_actions = env.action_space.shape[-1]
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

        # Configure components.
        memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)

        activation_map = { "relu" : tf.nn.relu, "leaky_relu" : U.lrelu, "tanh" :tf.nn.tanh}

        critic = Critic(layer_norm=layer_norm, hidden_sizes=kwargs["vf_size"], activation=activation_map[kwargs["activation_vf"]])
        actor = Actor(nb_actions, layer_norm=layer_norm, hidden_sizes=kwargs["policy_size"], activation=activation_map[kwargs["activation_policy"]])

        # Seed everything to make things reproducible.
        seed = all_seeds[e]
        print ("USING RANDOM SEED FOR EXPERIMENT", seed)
        seed = seed + 1000000 * rank
        logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
        tf.reset_default_graph()
        set_global_seeds(seed)
        env.seed(seed)
        if eval_env is not None:
            eval_env.seed(seed)



        # Disable logging for rank != 0 to avoid noise.
        if rank == 0:
            start_time = time.time()

        environment_name = env_id

        training.train(env=env, eval_env=eval_env, nb_epochs=nb_epochs, param_noise=param_noise,
            action_noise=action_noise, actor=actor, critic=critic, memory=memory, save_dir=save_dir, num_experiments=e, all_exp_eval_return=all_exp_eval_return, **kwargs)


        env.close()
        if eval_env is not None:
            eval_env.close()
        if rank == 0:
            logger.info('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='HalfCheetah-v1')
    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb_epochs', type=int, default=1000)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument("--policy_size", nargs="+", default=(64,64), type=int)
    parser.add_argument("--vf_size", nargs="+", default=(64,64), type=int)
    parser.add_argument("--activation_vf", type=str, default="relu")
    parser.add_argument("--activation_policy", type=str, default="relu")
    boolean_flag(parser, 'evaluation', default=True)
    parser.add_argument("--save_dir", type=str, default="trial_result")
    parser.add_argument("--num_experiments", type=int, default=5)
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    # Run actual script.
    run(**args)

