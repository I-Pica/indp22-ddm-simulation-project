import numpy as np
import matplotlib.pyplot as plt

def plot_rt_hist(hits, errs):
    '''
    Plot the reaction time distributions for the hit and error trials
    '''
    rt_hits_hist, rt_hits_bin_edges = np.histogram(hits)
    rt_errs_hist, rt_errs_bin_edges = np.histogram(errs)

    plt.figure(figsize=(5,5))
    plt.stairs(rt_hits_hist, rt_hits_bin_edges, label='hits')
    plt.stairs(rt_errs_hist, rt_errs_bin_edges, label='errors')
    plt.xlabel('Reaction time')
    plt.ylabel('Count')
    plt.legend(loc='upper right');
    return 0

def plot_psychometric(prob_a, mu_list, colors):
    '''
    Plot the psychometric curve for the given responses and drift
    parameters.
    '''
    # plt.figure(figsize=(5,5))
    plt.scatter(mu_list, prob_a, 80, color=colors, edgecolors='k')
    plt.xlabel('Drift')
    plt.ylabel('Porbability of a')
    plt.xlim([np.min(mu_list)*1.05, np.max(mu_list)*1.05])
    plt.ylim([0-0.05, 1+0.05]);
    return 0

def plot_quantile_prob_func(S_list, hits_list, errs_list, mu_list, prob_a):
    '''
    Plot the quantile probability function for the reaction times of a
    drift diffusion model.
    '''
    n_mu = len(mu_list)
    quantile_list = np.linspace(10,90,5)
    rt_quantiles = np.nan*np.ones((n_mu, len(quantile_list)))
    for mu_i in range(n_mu): # skip drift equal to zero
        um_i = n_mu-mu_i-1 # id of drift with same magnitude but opposite sign
        if mu_list[mu_i] < 0: # drift to b
            # errors for drifts to a and b, respectively
            rt_errs = np.concatenate((S_list[um_i][errs_list[um_i][0],1],\
                                      S_list[mu_i][errs_list[mu_i][0],1]))
            rt_quantiles[mu_i,:] = np.percentile(rt_errs, quantile_list)
        elif mu_list[mu_i] > 0: # drift to a
            # hits for drifts to a and b, respectively
            rt_hits = np.concatenate((S_list[mu_i][hits_list[mu_i][0],1],\
                                      S_list[um_i][hits_list[um_i][0],1]))
            rt_quantiles[mu_i,:] = np.percentile(rt_hits, quantile_list)

    plt.figure(figsize=(5,5))
    plt.plot(prob_a, rt_quantiles, marker='x', linestyle='')
    plt.xlabel('Probability of a')
    plt.ylabel('Reaction time quantile');
    return rt_quantiles

def plot_trajectories_and_RT(S, hits, errs, traj, mu, t, bins, colors):
   '''
   Plot the trajectories of two trials together with the reaction time distributions.
   '''
    
   trialExample = [hits[0][1], errs[0][1]]
   ranget = [np.sum(~np.isnan(traj[trialExample[0],:])), np.sum(~np.isnan(traj[trialExample[1],:]))]

   fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6,6))

   axes[0].hist(S[hits[0],1], bins, color=colors[0]);
   axes[0].set_ylabel('Correct Count');
   axes[0].scatter(t[ranget[0]], 1, 30,color='white', marker='o')

   axes[1].plot(t[0:ranget[0]], traj[trialExample[0],0:ranget[0]].T, color=colors[0], clip_on=False);    
   axes[1].plot(t[0:ranget[1]], traj[trialExample[1],0:ranget[1]].T, color=colors[1], clip_on=False);
   axes[1].plot(np.linspace(0,6,100),np.linspace(0,6,100)*mu, linestyle='--', linewidth=2,color='#B5BD89')
   axes[1].set_yticks([0]); 
   axes[1].set_yticklabels(['Starting Point (z)'], color='#729EA1');
   axes[1].set_ylim([-1, 1]);
   axes[1].spines['top'].set_color('#EC9192') 
   axes[1].spines['top'].set_linewidth(2)
   
   axes[2].spines['top'].set_color('#EC9192')
   axes[2].spines['top'].set_linewidth(2)
   axes[2].hist(S[errs[0],1], bins,color=colors[1]);
   axes[2].scatter(t[ranget[1]], 1, 30, color='white', marker='o')

   axes[2].invert_yaxis()   
   axes[2].set_xlim([0, 6]); 

   plt.subplots_adjust(hspace=0)
   plt.xlabel('Time from Onset (s)');
   plt.ylabel('Error Count');