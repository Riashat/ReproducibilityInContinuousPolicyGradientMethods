import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from numpy import genfromtxt
import pdb
from scipy import stats


plt.rcParams['text.usetex'] = True
eps = np.arange(0, 2e6, 2000)


#HalfCheetah Value Size
hs_400_300 = np.load('/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baselines_ddpg_results/Results/HalfCheetah_Policy_Activation_100_50_all_exp_rewards.npy')
hs_64_64 = np.load('/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baselines_ddpg_results/Results/HalfCheetah_Policy_Act_Relu_all_exp_rewards.npy')
hs_100_50 = np.load('/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baselines_ddpg_results/Results/HalfCheetah_VF_Size_100_50_all_exp_rewards.npy')



mean_hs_64_64 = np.mean(hs_64_64, axis=1)
mean_hs_400_300 = np.mean(hs_400_300, axis=1)
mean_hs_100_50 = np.mean(hs_100_50, axis=1)

std_hs_64_64 = np.std(hs_64_64, axis=1)
std_hs_400_300 = np.std(hs_400_300, axis=1)
std_hs_100_50 = np.std(hs_100_50, axis=1)



last_ho_64_64 = mean_hs_64_64[-1]
last_error_ho_64_64 = stats.sem(hs_64_64[-1, :], axis=None, ddof=0)

print ("last_ho_64_64", last_ho_64_64)
print ("last_error_ho_64_64", last_error_ho_64_64)

last_ho_100_50 = mean_hs_100_50[-1]
last_error_ho_100_50 = stats.sem(hs_100_50[-1, :], axis=None, ddof=0)

print ("last_ho_100_50",last_ho_100_50)
print ("last_error_ho_100_50", last_error_ho_100_50)


last_ho_400_300 = mean_hs_400_300[-1]
last_error_ho_400_300 = stats.sem(hs_400_300[-1, :], axis=None, ddof=0)

print ("last_ho_400_300", last_ho_400_300)
print ("last_error_ho_400_300",last_error_ho_400_300)





#Hopper Value Size
ho_400_300 = np.load('/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baselines_ddpg_results/Results/Hopper_Policy_Size_400_300_all_exp_rewards.npy')
ho_64_64 = np.load('/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baselines_ddpg_results/Results/Hopper_Policy_Activation_Relu_all_exp_rewards.npy')
ho_100_50  = np.load('/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baselines_ddpg_results/Results/Hopper_VF_Size_100_50_all_exp_rewards.npy')


mean_ho_64_64 = np.mean(ho_64_64, axis=1)
mean_ho_400_300 = np.mean(ho_400_300, axis=1)
mean_ho_100_50 = np.mean(ho_100_50, axis=1)

std_ho_64_64 = np.std(ho_64_64, axis=1)
std_ho_400_300 = np.std(ho_400_300, axis=1)
std_ho_100_50 = np.std(ho_100_50, axis=1)



last_ho_64_64 = mean_ho_64_64[-1]
last_error_ho_64_64 = stats.sem(ho_64_64[-1, :], axis=None, ddof=0)

print ("last_ho_64_64", last_ho_64_64)
print ("last_error_ho_64_64", last_error_ho_64_64)

last_ho_100_50 = mean_ho_100_50[-1]
last_error_ho_100_50 = stats.sem(ho_100_50[-1, :], axis=None, ddof=0)

print ("last_ho_100_50",last_ho_100_50)
print ("last_error_ho_100_50", last_error_ho_100_50)


last_ho_400_300 = mean_ho_400_300[-1]
last_error_ho_400_300 = stats.sem(ho_400_300[-1, :], axis=None, ddof=0)

print ("last_ho_400_300", last_ho_400_300)
print ("last_error_ho_400_300",last_error_ho_400_300)




# def halfcheetah_plot(stats1, stats2, stats3, smoothing_window=5, noshow=False):
#     ## Figure 1
#     fig = plt.figure(figsize=(70, 40))
#     rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
#     rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
#     rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()

#     cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, color = "red", linewidth=2.5, label="Critic Network Size = 64 x 64")    
#     plt.fill_between( eps, rewards_smoothed_1 + std_hs_64_64,   rewards_smoothed_1 - std_hs_64_64, alpha=0.2, edgecolor='red', facecolor='red')

#     cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, color = "blue", linewidth=2.5, label="Critic Network Size = 100 x 50 x 25" )  
#     plt.fill_between( eps, rewards_smoothed_2 + std_hs_100_50,   rewards_smoothed_2 - std_hs_100_50, alpha=0.2, edgecolor='blue', facecolor='blue')

#     cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, color = "black", linewidth=2.5, label="Critic Network Size = 400 x 300" )  
#     plt.fill_between( eps, rewards_smoothed_3 + std_hs_400_300,   rewards_smoothed_3 - std_hs_400_300, alpha=0.2, edgecolor='black', facecolor='black')

#     plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3], fontsize=22)
#     plt.xlabel("Number of Iterations", fontsize=26)
#     plt.ylabel("Average Return", fontsize=26)
#     plt.title("DDPG with HalfCheetah Environment, Critic Network Size", fontsize=30)
  
#     plt.show()

#     fig.savefig('ddpg_halfcheetah_value_function_size.png')

    
#     return fig


def halfcheetah_plot(stats1, stats2, stats3,  smoothing_window=5, noshow=False):
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

    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, color = "#1f77b4", linewidth=2.5, label="Critic Network Size = 64 x 64")    
    plt.fill_between( eps, rewards_smoothed_1 + std_hs_64_64,   rewards_smoothed_1 - std_hs_64_64, alpha=0.2, edgecolor='#1f77b4', facecolor='#1f77b4')

    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, color = "#ff7f0e", linewidth=2.5, label="Critic Network Size = 100 x 50 x 25" )  
    plt.fill_between( eps, rewards_smoothed_2 + std_hs_100_50,   rewards_smoothed_2 - std_hs_100_50, alpha=0.2, edgecolor='#ff7f0e', facecolor='#ff7f0e')

    cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, color = "#d62728", linewidth=2.5, label="Critic Network Size = 400 x 300" )  
    plt.fill_between( eps, rewards_smoothed_3 + std_hs_400_300,   rewards_smoothed_3 - std_hs_400_300, alpha=0.2, edgecolor='#d62728', facecolor='#d62728')

    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3],  loc='lower right', prop={'size' : 16})
    plt.xlabel("Timesteps", **axis_font)
    plt.ylabel("Average Returns", **axis_font)
    plt.title("DDPG with HalfCheetah Environment, Critic Network Size", **axis_font)
  
    plt.show()
    
    fig.savefig('ddpg_halfcheetah_value_function_size.png')
    return fig




# def hopper_plot(stats1, stats2, stats3, smoothing_window=5, noshow=False):
#     ## Figure 1
#     fig = plt.figure(figsize=(70, 40))
#     rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
#     rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
#     rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()

#     cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, color = "red", linewidth=2.5, label="Critic Network Size = 64 x 64")    
#     plt.fill_between( eps, rewards_smoothed_1 + std_ho_64_64,   rewards_smoothed_1 - std_ho_64_64, alpha=0.2, edgecolor='red', facecolor='red')

#     cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, color = "blue", linewidth=2.5, label="Critic Network Size = 100 x 50 x 25" )  
#     plt.fill_between( eps, rewards_smoothed_2 + std_ho_100_50,   rewards_smoothed_2 - std_ho_100_50, alpha=0.2, edgecolor='blue', facecolor='blue')

#     cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, color = "black", linewidth=2.5, label="Critic Network Size = 400 x 300" )  
#     plt.fill_between( eps, rewards_smoothed_3 + std_ho_400_300,   rewards_smoothed_3 - std_ho_400_300, alpha=0.2, edgecolor='black', facecolor='black')

#     plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3],fontsize=22)
#     plt.xlabel("Number of Iterations", fontsize=26)
#     plt.ylabel("Average Return", fontsize=26)
#     plt.title("DDPG with Hopper Environment, Critic Network Size", fontsize=30)
  
#     plt.show()
    
#     fig.savefig('ddpg_hopper_value_function_size.png')

#     return fig



def hopper_plot(stats1, stats2, stats3,  smoothing_window=5, noshow=False):
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

    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, color = "#1f77b4", linewidth=2.5, label="Critic Network Size = 64 x 64")    
    plt.fill_between( eps, rewards_smoothed_1 + std_ho_64_64,   rewards_smoothed_1 - std_ho_64_64, alpha=0.2, edgecolor='#1f77b4', facecolor='#1f77b4')

    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, color = "#ff7f0e", linewidth=2.5, label="Critic Network Size = 100 x 50 x 25" )  
    plt.fill_between( eps, rewards_smoothed_2 + std_ho_100_50,   rewards_smoothed_2 - std_ho_100_50, alpha=0.2, edgecolor='#ff7f0e', facecolor='#ff7f0e')

    cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, color = "#d62728", linewidth=2.5, label="Critic Network Size = 400 x 300" )  
    plt.fill_between( eps, rewards_smoothed_3 + std_ho_400_300,   rewards_smoothed_3 - std_ho_400_300, alpha=0.2, edgecolor='#d62728', facecolor='#d62728')

    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3],  loc='lower right', prop={'size' : 16})
    plt.xlabel("Timesteps", **axis_font)
    plt.ylabel("Average Returns", **axis_font)
    plt.title("DDPG with Hopper Environment, Critic Network Size", **axis_font)
  
    plt.show()
    
    fig.savefig('ddpg_hopper_value_function_size.png')
    return fig











def main():
   hopper_plot(mean_ho_64_64, mean_ho_100_50, mean_ho_400_300)

   halfcheetah_plot(mean_hs_64_64, mean_hs_100_50, mean_hs_400_300)


if __name__ == '__main__':
    main()