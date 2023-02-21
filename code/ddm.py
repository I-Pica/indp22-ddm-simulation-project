import numpy as np
import matplotlib.pyplot as plt

def sim_ddm(mu=0.5, theta=1, b=0, z=0, sigma=1, n_trials=1000, dt=.001, T=10):
   '''
   Perform a simulation with a drift diffusion model for n_trials.
   '''
   # time array and prealocate results
   t = np.arange(0, T, dt)
   n_t = t.size
   S = -np.ones((n_trials,2))
   traj = np.zeros((n_trials, n_t))

   # compute time varying bounds for b != 0
   thetas = theta/(1+(b/dt*np.linspace(0, 1, n_t)))
   
   for tr in range(n_trials):
      dW   = np.random.randn(n_t,1) # noise vector
      dx   = mu*dt + sigma*dW*np.sqrt(dt)
      dx[0] += z # starting point
      x    = np.cumsum(dx,0)
      traj[tr,:] = np.squeeze(x)
  
      for ti in range(n_t):
         if x[ti] >= thetas[ti]:
            S[tr,:] = [1, t[ti]]                                         
            break
         elif x[ti] <= -thetas[ti]:
            S[tr,:] = [0, t[ti]]         
            break
      
      traj[tr, ti:n_t] = np.nan
        
   # Get the hit and error trials
   if np.sign(mu)>0:
      hits = np.where(S[:,0]==1)
      errs = np.where(S[:,0]==0)
   else:
      hits = np.where(S[:,0]==0)
      errs = np.where(S[:,0]==1)
   return S, hits, errs, thetas, traj

def race_sim():
    '''
    Perform a simulation with a race diffusion model for n_trials.
    '''
    return 0

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

def calc_psychometric(S_list, mu_list):
    ''''
    Compute the response ratio as a function of the drift.
    '''
    n_mu = len(mu_list)
    a_rate = np.zeros((n_mu,))
    b_rate = np.zeros((n_mu,))
    for i in range(n_mu):
        a_rate[i] = np.sum(S_list[i]==1)
        b_rate[i] = np.sum(S_list[i]==0)
    prob_a = np.divide(a_rate, a_rate+b_rate) # probability of a
    return prob_a

def plot_psychometric(prob_a, mu_list):
   '''
   Plot the psychometric curve for the given responses and drift
   parameters.
   '''
   plt.figure(figsize=(5,5))
   plt.scatter(mu_list, prob_a, color='k')
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

def plot_trajectories_and_RT(S, hits, errs, traj, t, bins):
   '''
   Plot the trajectories of two trials together with the reaction time distributions.
   '''
    
   trialExample = [hits[0][1], errs[0][1]]
   ranget = [np.sum(~np.isnan(traj[trialExample[0],:])), np.sum(~np.isnan(traj[trialExample[1],:]))]

   fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8,8))

   #axes[0].subplots_adjust(hspace=0)
   axes[0].hist(S[hits[0],1], bins, color='teal');

   axes[1].plot(t[0:ranget[0]], traj[trialExample[0],0:ranget[0]].T,color='teal');    
   axes[1].plot(t[0:ranget[1]], traj[trialExample[1],0:ranget[1]].T,color='orange');
   axes[1].set_ylim([-1, 1]); axes[1].set_yticks([0])

   axes[2].hist(S[errs[0],1], bins,color='orange');
   axes[2].invert_yaxis()    
   plt.subplots_adjust(hspace=0)
   plt.xlabel('Time from Onset (s)');
   plt.ylabel('Trial Count');