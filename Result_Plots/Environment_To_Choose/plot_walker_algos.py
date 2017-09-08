import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from numpy import genfromtxt
import pdb

# episodes is the total number of rows of arrays heretrpo_1

eps = np.arange(401)

plt.rcParams['text.usetex'] = True

"""
TRPO Results
"""
trpo_1 = pd.read_csv("/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baseline_trpo_results/Env_Choose/Walker/exp_1/progress.csv")
trpo_2 = pd.read_csv("/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baseline_trpo_results/Env_Choose/Walker/exp_2/progress.csv")
trpo_3 = pd.read_csv("/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baseline_trpo_results/Env_Choose/Walker/exp_3/progress.csv")
trpo_4 = pd.read_csv("/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baseline_trpo_results/Env_Choose/Walker/exp_4/progress.csv")
trpo_5 = pd.read_csv("/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baseline_trpo_results/Env_Choose/Walker/exp_5/progress.csv")


trpo_1_timesteps = np.array(trpo_1["TimestepsSoFar"])
trpo_1_episode_reward = np.array(trpo_1["EpRewMean"])
trpo_1_episodes_so_far = np.array(trpo_1["EpisodesSoFar"])

trpo_2_timesteps = np.array(trpo_2["TimestepsSoFar"])
trpo_2_episode_reward = np.array(trpo_2["EpRewMean"])
trpo_2_episodes_so_far = np.array(trpo_2["EpisodesSoFar"])

trpo_3_timesteps = np.array(trpo_3["TimestepsSoFar"])
trpo_3_episode_reward = np.array(trpo_3["EpRewMean"])
trpo_3_episodes_so_far = np.array(trpo_3["EpisodesSoFar"])


trpo_4_timesteps = np.array(trpo_4["TimestepsSoFar"])
trpo_4_episode_reward = np.array(trpo_4["EpRewMean"])
trpo_4_episodes_so_far = np.array(trpo_4["EpisodesSoFar"])


trpo_5_timesteps = np.array(trpo_5["TimestepsSoFar"])
trpo_5_episode_reward = np.array(trpo_5["EpRewMean"])
trpo_5_episodes_so_far = np.array(trpo_5["EpisodesSoFar"])



trpo_time_steps = np.column_stack((trpo_1_timesteps, trpo_2_timesteps, trpo_3_timesteps, trpo_4_timesteps, trpo_5_timesteps))
mean_trpo_time_steps = np.mean(trpo_time_steps, axis=1)

trpo_episode_reward = np.column_stack((trpo_1_episode_reward, trpo_2_episode_reward, trpo_3_episode_reward, trpo_4_episode_reward, trpo_5_episode_reward))
mean_trpo_episode_reward = np.mean(trpo_episode_reward, axis=1)
std_trpo_episode_reward = np.std(trpo_episode_reward, axis=1)

trpo_episodes_so_far = np.column_stack((trpo_1_episodes_so_far, trpo_2_episodes_so_far, trpo_3_episodes_so_far, trpo_4_episodes_so_far, trpo_5_episodes_so_far))
mean_trpo_episodes_so_far = np.mean(trpo_episodes_so_far, axis=1)



np.save('./Result_Arrays/trpo_walker.npy', trpo_episode_reward)



"""
PPO Results
"""
ppo_1 = pd.read_csv("/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baseline_ppo_results/Env_Choose/Walker/exp_1/progress.csv")
ppo_2 = pd.read_csv("/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baseline_ppo_results/Env_Choose/Walker/exp_2/progress.csv")
ppo_3 = pd.read_csv("/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baseline_ppo_results/Env_Choose/Walker/exp_3/progress.csv")
ppo_4 = pd.read_csv("/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baseline_ppo_results/Env_Choose/Walker/exp_4/progress.csv")
ppo_5 = pd.read_csv("/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baseline_ppo_results/Env_Choose/Walker/exp_5/progress.csv")


ppo_1_timesteps = np.array(ppo_1["TimestepsSoFar"])
ppo_1_episode_reward = np.array(ppo_1["EpRewMean"])
ppo_1_episodes_so_far = np.array(ppo_1["EpisodesSoFar"])

ppo_2_timesteps = np.array(ppo_2["TimestepsSoFar"])
ppo_2_episode_reward = np.array(ppo_2["EpRewMean"])
ppo_2_episodes_so_far = np.array(ppo_2["EpisodesSoFar"])

ppo_3_timesteps = np.array(ppo_3["TimestepsSoFar"])
ppo_3_timesteps = ppo_3_timesteps[0:977]
ppo_3_episode_reward = np.array(ppo_3["EpRewMean"])
ppo_3_episode_reward = ppo_3_episode_reward[0:977]
ppo_3_episodes_so_far = np.array(ppo_3["EpisodesSoFar"])
ppo_3_episodes_so_far = ppo_3_episodes_so_far[0:977]



ppo_4_timesteps = np.array(ppo_4["TimestepsSoFar"])
ppo_4_episode_reward = np.array(ppo_4["EpRewMean"])
ppo_4_episodes_so_far = np.array(ppo_4["EpisodesSoFar"])

ppo_5_timesteps = np.array(ppo_5["TimestepsSoFar"])
ppo_5_episode_reward = np.array(ppo_5["EpRewMean"])
ppo_5_episodes_so_far = np.array(ppo_5["EpisodesSoFar"])



ppo_time_steps = np.column_stack((ppo_1_timesteps, ppo_2_timesteps, ppo_3_timesteps, ppo_4_timesteps, ppo_5_timesteps))
mean_ppo_time_steps = np.mean(ppo_time_steps, axis=1)

ppo_episode_reward = np.column_stack((ppo_1_episode_reward, ppo_2_episode_reward, ppo_3_episode_reward, ppo_4_episode_reward, ppo_5_episode_reward))
mean_ppo_episode_reward = np.mean(ppo_episode_reward, axis=1)
std_ppo_episode_reward = np.std(ppo_episode_reward, axis=1)

ppo_episodes_so_far = np.column_stack((ppo_1_episodes_so_far, ppo_2_episodes_so_far, ppo_3_episodes_so_far, ppo_4_episodes_so_far, ppo_5_episodes_so_far))
mean_ppo_episodes_so_far = np.mean(ppo_episodes_so_far, axis=1)


np.save('./Result_Arrays/ppo_walker.npy', ppo_episode_reward)


"""
DDPG Results
"""
ddpg_episode_reward = np.load('/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baselines_ddpg_results/Results/Env_Exp_DDPG_Walker_all_exp_rewards.npy')
mean_ddpg_episode_reward = np.mean(ddpg_episode_reward, axis=1)
std_ddpg_episode_reward = np.std(ddpg_episode_reward, axis=1)

# mean_ddpg_episode_reward = mean_ddpg_episode_reward[0:977]
# std_ddpg_episode_reward = std_ddpg_episode_reward[0:977]

ddpg_time_steps = np.arange(0, 2e6, 2000)
# ddpg_time_steps = ddpg_time_steps[0:977]

np.save('./Result_Arrays/ddpg_walker.npy', ddpg_episode_reward)




"""
ACKTR Results
"""


acktr_1 = pd.read_csv("/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baseline_acktr_results/Env_Choose/Walker/exp_1/progress.csv")
acktr_2 = pd.read_csv("/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baseline_acktr_results/Env_Choose/Walker/exp_2/progress.csv")
acktr_3 = pd.read_csv("/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baseline_acktr_results/Env_Choose/Walker/exp_3/progress.csv")
acktr_4 = pd.read_csv("/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baseline_acktr_results/Env_Choose/Walker/exp_4/progress.csv")
acktr_5 = pd.read_csv("/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baseline_acktr_results/Env_Choose/Walker/exp_5/progress.csv")


acktr_1_episode_reward = np.array(acktr_1["EpRewMean"])
acktr_1_episode_reward = acktr_1_episode_reward[0:730]
acktr_2_episode_reward = np.array(acktr_2["EpRewMean"])
acktr_2_episode_reward = acktr_2_episode_reward[0:730]
acktr_3_episode_reward = np.array(acktr_3["EpRewMean"])
acktr_3_episode_reward = acktr_3_episode_reward[0:730]
acktr_4_episode_reward = np.array(acktr_4["EpRewMean"])
acktr_4_episode_reward = acktr_4_episode_reward[0:730]
acktr_5_episode_reward = np.array(acktr_5["EpRewMean"])
acktr_5_episode_reward = acktr_5_episode_reward[0:730]




# acktr_time_steps = np.column_stack((acktr_1_timesteps, acktr_2_timesteps, acktr_3_timesteps, acktr_4_timesteps, acktr_5_timesteps))
# mean_acktr_time_steps = np.mean(acktr_time_steps, axis=1)

acktr_episode_reward = np.column_stack((acktr_1_episode_reward, acktr_2_episode_reward, acktr_3_episode_reward, acktr_4_episode_reward, acktr_5_episode_reward))
mean_acktr_episode_reward = np.mean(acktr_episode_reward, axis=1)
std_acktr_episode_reward = np.std(acktr_episode_reward, axis=1)

# acktr_episodes_so_far = np.column_stack((acktr_1_episodes_so_far, acktr_2_episodes_so_far, acktr_3_episodes_so_far, acktr_4_episodes_so_far, acktr_5_episodes_so_far))
# mean_acktr_episodes_so_far = np.mean(acktr_episodes_so_far, axis=1)



acktr_time_steps = np.arange(0, 2e6, 2700)
acktr_time_steps = acktr_time_steps[0:730]


np.save('./Result_Arrays/acktr_walker.npy', acktr_episode_reward)



pdb.set_trace()



# def double_plot(stats1, stats2, stats3, stats4, smoothing_window=5, noshow=False):
#     ## Figure 1
#     fig = plt.figure(figsize=(70, 40))
#     rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
#     rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
#     rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()
#     rewards_smoothed_4 = pd.Series(stats4).rolling(smoothing_window, min_periods=smoothing_window).mean()

#     cum_rwd_1, = plt.plot(mean_trpo_time_steps, rewards_smoothed_1, color = "red", linewidth=2.5, label="TRPO")    
#     plt.fill_between( mean_trpo_time_steps, rewards_smoothed_1 + std_trpo_episode_reward,   rewards_smoothed_1 - std_trpo_episode_reward, alpha=0.2, edgecolor='red', facecolor='red')

#     cum_rwd_2, = plt.plot(mean_ppo_time_steps, rewards_smoothed_2, color = "blue", linewidth=2.5, label="PPO" )  
#     plt.fill_between( mean_ppo_time_steps, rewards_smoothed_2 + std_ppo_episode_reward,   rewards_smoothed_2 - std_ppo_episode_reward, alpha=0.2, edgecolor='blue', facecolor='blue')

#     cum_rwd_3, = plt.plot(ddpg_time_steps, rewards_smoothed_3, color = "black", linewidth=2.5, label="DDPG" )  
#     plt.fill_between( ddpg_time_steps, rewards_smoothed_3 + std_ddpg_episode_reward,   rewards_smoothed_3 - std_ddpg_episode_reward, alpha=0.2, edgecolor='black', facecolor='black')

#     cum_rwd_4, = plt.plot(acktr_time_steps, rewards_smoothed_4, color = "green", linewidth=2.5, label="ACKTR" )  
#     plt.fill_between( acktr_time_steps, rewards_smoothed_4 + std_acktr_episode_reward,   rewards_smoothed_4 - std_acktr_episode_reward, alpha=0.2, edgecolor='green', facecolor='green')

#     plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3, cum_rwd_4], fontsize=28)
#     plt.xlabel("Number of Time Steps", fontsize=26)
#     plt.ylabel("Average Return", fontsize=26)
#     plt.title("Walker Environment", fontsize=30)
  
#     plt.show()

#     fig.savefig('Walker_Env_Algos.png')
#     return fig



def double_plot(stats1, stats2, stats3, stats4, smoothing_window=1, noshow=False):
    ## Figure 1
    fig = plt.figure(figsize=(16, 8))
    ax = plt.subplot()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(22)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(20)
    axis_font = {'fontname':'Arial', 'size':'28'}

    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_4 = pd.Series(stats4).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd_1, = plt.plot(mean_trpo_time_steps, rewards_smoothed_1, color = "#1f77b4", linewidth=2.5, label="TRPO")    
    plt.fill_between( mean_trpo_time_steps, rewards_smoothed_1 + std_trpo_episode_reward,   rewards_smoothed_1 - std_trpo_episode_reward, alpha=0.2, edgecolor='#1f77b4', facecolor='#1f77b4')

    cum_rwd_2, = plt.plot(mean_ppo_time_steps, rewards_smoothed_2, color = "#ff7f0e", linewidth=2.5, label="PPO" )  
    plt.fill_between( mean_ppo_time_steps, rewards_smoothed_2 + std_ppo_episode_reward,   rewards_smoothed_2 - std_ppo_episode_reward, alpha=0.2, edgecolor='#ff7f0e', facecolor='#ff7f0e')

    cum_rwd_3, = plt.plot(ddpg_time_steps, rewards_smoothed_3, color = "#d62728", linewidth=2.5, label="DDPG" )  
    plt.fill_between( ddpg_time_steps, rewards_smoothed_3 + std_ddpg_episode_reward,   rewards_smoothed_3 - std_ddpg_episode_reward, alpha=0.2, edgecolor='#d62728', facecolor='#d62728')

    cum_rwd_4, = plt.plot(acktr_time_steps, rewards_smoothed_4, color = "#9467bd", linewidth=2.5, label="ACKTR" )  
    plt.fill_between( acktr_time_steps, rewards_smoothed_4 + std_acktr_episode_reward,   rewards_smoothed_4 - std_acktr_episode_reward, alpha=0.2, edgecolor='#9467bd', facecolor='#9467bd')

    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3, cum_rwd_4], loc='lower right', prop={'size' : 16})

    plt.xlabel("Timesteps", **axis_font)
    plt.ylabel("Average Return", **axis_font)
    plt.title("Walker Environment", **axis_font)
  
    plt.show()

    fig.savefig('Walker_Env_Algos.png')
    
    return fig








def main():
   double_plot(mean_trpo_episode_reward, mean_ppo_episode_reward, mean_ddpg_episode_reward, mean_acktr_episode_reward)


if __name__ == '__main__':
    main()