import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from numpy import genfromtxt
import pdb
from scipy import stats


eps = np.arange(0, 2e6, 2000)

plt.rcParams['text.usetex'] = True


#HalfCheetah Policy Activations
hs_leaky_relu = np.load('/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baselines_ddpg_results/Results/HalfCheetah_Policy_Act_Leaky_Relu_all_exp_rewards.npy')
hs_relu = np.load('/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baselines_ddpg_results/Results/HalfCheetah_Policy_Act_Relu_all_exp_rewards.npy')
hs_tanh = np.load('/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baselines_ddpg_results/Results/HalfCheetah_Policy_Act_TanH_all_exp_rewards.npy')

mean_hs_leaky = np.mean(hs_leaky_relu, axis=1)
mean_hs_relu = np.mean(hs_relu, axis=1)
mean_hs_tanh = np.mean(hs_tanh, axis=1)

std_hs_leaky = np.std(hs_leaky_relu, axis=1)
std_hs_relu = np.std(hs_relu, axis=1)
std_hs_tanh = np.std(hs_tanh, axis=1)


last_hs_leaky = mean_hs_leaky[-1]
last_error_hs_leaky = stats.sem(hs_leaky_relu[-1, :], axis=None, ddof=0)

print ("last_hs_leaky", last_hs_leaky)
print ("last_error_hs_leaky", last_error_hs_leaky)


last_hs_relu = mean_hs_relu[-1]
last_error_hs_relu = stats.sem(hs_relu[-1, :], axis=None, ddof=0)

print ("last_hs_relu", last_hs_relu)
print ("last_error_hs_relu", last_error_hs_relu)



last_hs_tanh = mean_hs_tanh[-1]
last_error_hs_tanh = stats.sem(hs_tanh[-1, :], axis=None, ddof=0)

print ("last_hs_tanh", last_hs_tanh)
print ("last_error_hs_tanh", last_error_hs_tanh)





#Hopper Policy Activations
ho_leaky_relu = np.load('/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baselines_ddpg_results/Results/Hopper_Policy_Activation_Leaky_Relu_all_exp_rewards.npy')
ho_relu = np.load('/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baselines_ddpg_results/Results/Hopper_Policy_Activation_Relu_all_exp_rewards.npy')
ho_tanh = np.load('/Users/Riashat/Documents/PhD_Research/OpenAIBaselines/ReproducibilityML/Results/rllab_results/baselines_ddpg_results/Results/Hopper_Policy_Activation_TanH_all_exp_rewards.npy')

mean_ho_leaky = np.mean(ho_leaky_relu, axis=1)
mean_ho_relu = np.mean(ho_relu, axis=1)
mean_ho_tanh = np.mean(ho_tanh, axis=1)

std_ho_leaky = np.std(ho_leaky_relu, axis=1)
std_ho_relu = np.std(ho_relu, axis=1)
std_ho_tanh = np.std(ho_tanh, axis=1)


last_ho_leaky = mean_ho_leaky[-1]
last_error_ho_leaky = stats.sem(ho_leaky_relu[-1, :], axis=None, ddof=0)

print ("last_ho_leaky", last_ho_leaky)
print ("last_error_ho_leaky", last_error_ho_leaky)


last_ho_relu = mean_ho_relu[-1]
last_error_ho_relu = stats.sem(ho_relu[-1, :], axis=None, ddof=0)

print ("last_ho_relu", last_ho_relu)
print ("last_error_ho_relu", last_error_ho_relu)



last_ho_tanh = mean_ho_tanh[-1]
last_error_ho_tanh = stats.sem(ho_tanh[-1, :], axis=None, ddof=0)

print ("last_ho_tanh", last_ho_tanh)
print ("last_error_ho_tanh", last_error_ho_tanh)





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

    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, color = "#1f77b4", linewidth=2.5, label="Policy Network Activation = ReLU")    
    plt.fill_between( eps, rewards_smoothed_1 + std_ho_relu,   rewards_smoothed_1 - std_ho_relu, alpha=0.2, edgecolor='#1f77b4', facecolor='#1f77b4')

    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, color = "#ff7f0e", linewidth=2.5, label="Policy Network Activation = TanH" )  
    plt.fill_between( eps, rewards_smoothed_2 + std_ho_tanh,   rewards_smoothed_2 - std_ho_tanh, alpha=0.2, edgecolor='#ff7f0e', facecolor='#ff7f0e')

    cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, color = "#d62728", linewidth=2.5, label="Policy Network Activation = Leaky ReLU" )  
    plt.fill_between( eps, rewards_smoothed_3 + std_ho_leaky,   rewards_smoothed_3 - std_ho_leaky, alpha=0.2, edgecolor='#d62728', facecolor='#d62728')

    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3],  loc='lower right', prop={'size' : 16})
    plt.xlabel("Timesteps",**axis_font)
    plt.ylabel("Average Returns", **axis_font)
    plt.title("DDPG with Hopper Environment - Actor Network Activations", **axis_font)
  
    plt.show()

    fig.savefig('ddpg_hopper_policy_activations.png')
    
    return fig



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

    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, color = "#1f77b4", linewidth=2.5, label="Policy Network Activation = ReLU")    
    plt.fill_between( eps, rewards_smoothed_1 + std_hs_relu,   rewards_smoothed_1 - std_hs_relu, alpha=0.2, edgecolor='#1f77b4', facecolor='#1f77b4')

    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, color = "#ff7f0e", linewidth=2.5, label="Policy Network Activation = TanH" )  
    plt.fill_between( eps, rewards_smoothed_2 + std_hs_tanh,   rewards_smoothed_2 - std_hs_tanh, alpha=0.2, edgecolor='#ff7f0e', facecolor='#ff7f0e')

    cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, color = "#d62728", linewidth=2.5, label="Policy Network Activation = Leaky ReLU" )  
    plt.fill_between( eps, rewards_smoothed_3 + std_hs_leaky,   rewards_smoothed_3 - std_hs_leaky, alpha=0.2, edgecolor='#d62728', facecolor='#d62728')

    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3],  loc='lower right', prop={'size' : 16})
    plt.xlabel("Timesteps",**axis_font)
    plt.ylabel("Average Returns", **axis_font)
    plt.title("DDPG with HalfCheetah Environment - Actor Network Activations", **axis_font)
  
    plt.show()

    fig.savefig('ddpg_halfcheetah_policy_activations.png')
    
    return fig




  


def main():
   hopper_plot(mean_ho_relu, mean_ho_tanh, mean_ho_leaky)

   halfcheetah_plot(mean_hs_relu, mean_hs_tanh, mean_hs_leaky)

if __name__ == '__main__':
    main()