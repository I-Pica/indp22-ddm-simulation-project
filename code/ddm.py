import numpy as np
import matplotlib.pyplot as plt

def drift_diff(mu, theta, z, sigma, dt, T, clamp_x = [], racer=False):
    '''
    Simulate a (bounded) drift diffusion process.
    '''
    # time array and pre-allocate trajectories
    t = np.arange(0, T, dt)
    n_t = t.size
    traj = np.nan*np.ones((n_t,))

    dW = np.random.randn(n_t,1) # noise vector
    dx = mu*dt + sigma*dW*np.sqrt(dt)
    dx[0] += z # starting point
    x  = np.cumsum(dx,0)
    if len(clamp_x)==2: 
        # Note: clamping destroys noise
        x[clamp_x[0][-1]:] -= x[clamp_x[0][-1]] # correct post-clamping signal
        x[clamp_x[0]] = clamp_x[1] # clamp signal
    S = -np.ones((2,))
    for ti in range(n_t):
        if x[ti] >= theta[ti]:
            S = [1, t[ti]]
            break
        elif x[ti] <= -theta[ti] and (not racer):
            S = [0, t[ti]]
            break

    traj[:ti] = np.squeeze(x[:ti])
    return S, traj, ti

def sim_ddm(mu=0.5, theta=1, z=0, sigma=1, n_trials=1000, dt=.001, T=10, clamp_x=[]):
    '''
    Perform a simulation with a drift diffusion model for n_trials.
    '''
    # time array and pre-alocate results
    t = np.arange(0, T, dt)
    n_t = t.size
    S = -np.ones((n_trials, 2))
    traj = np.empty((n_trials, n_t))
   
    for tr in range(n_trials):
        S[tr,:], traj[tr,:], _ = drift_diff(mu, theta, z, sigma, dt, T, clamp_x)
    return S, traj

def sim_race(mu, theta, z, sigma, n_trials=1000, dt=.001, T=10, clamp_x=[]):
    '''
    Perform a simulation with a race diffusion model for n_trials.
    The race diffusion simulation has two racers (a and b).
    '''
    # time array and pre-alocate results
    t = np.arange(0, T, dt)
    n_t = t.size
    S = -np.ones((n_trials, 2))
    C = np.nan*np.ones((n_trials,)) # confidence
    traj = np.empty((2, n_trials, n_t)) # trajectories for the two racers

    ti = [0, 0]
    for tr in range(n_trials):
        A, traj[0,tr,:], ti[0] = drift_diff(mu[0], theta[0], z[0], sigma[0], dt, T, clamp_x, racer=True)
        B, traj[1,tr,:], ti[1] = drift_diff(mu[1], theta[1], z[1], sigma[1], dt, T, clamp_x, racer=True)
        # Get the reaction time of the racer that crossed their bound first
        S[tr,1] = np.max((np.min((A[1], B[1])), -A[1]*B[1]))
        i = np.argmax((A[1]==S[tr,1], B[1]==S[tr,1])) # the index of the winner
        j = (i+1) % 2 # the index of the loser
        S[tr,0] = i
        # Get the confidence, which is theta[loser] - position[loser] @ winner_time
        # Note: I don't think this is a good proxy for confidence.
        C[tr] = theta[j]-traj[j,tr,ti[i]]
    return S, C, traj

def calc_hits_errs(S, mu):
    # Get the hit and error trials
    if np.sign(mu)>0:
        hits = np.where(S[:,0]==1)
        errs = np.where(S[:,0]==0)
    else:
        hits = np.where(S[:,0]==0)
        errs = np.where(S[:,0]==1)
    return hits, errs

def calc_time_var_bound(theta, b, dt, n_t):
    # compute time varying bounds for b != 0
    thetas = theta/(1+(b/dt*np.linspace(0, 1, n_t)))
    return thetas

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