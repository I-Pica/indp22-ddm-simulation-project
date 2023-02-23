import numpy as np
import matplotlib.pyplot as plt

def drift_diff(mu, theta, z, sigma, dt, T, clamp_x = [], racer=False, seed=None):
    '''
    Simulate a (bounded) drift diffusion process.
    '''
    np.random.seed(seed)
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

def sim_ddm(mu=0.5, theta=1, z=0, sigma=1, n_trials=1000, dt=.001, T=10, clamp_x=[], seed=None):
    '''
    Perform a simulation with a drift diffusion model for n_trials.
    '''
    np.random.seed(seed)
    # time array and pre-alocate results
    t = np.arange(0, T, dt)
    n_t = t.size
    S = -np.ones((n_trials, 2))
    traj = np.empty((n_trials, n_t))
   
    for tr in range(n_trials):
        S[tr,:], traj[tr,:], _ = drift_diff(mu, theta, z, sigma, dt, T, clamp_x)
    return S, traj

def sim_race(mu, theta, z, sigma, n_trials=1000, dt=.001, T=10, clamp_x=[], seed=None):
    '''
    Perform a simulation with a race diffusion model for n_trials.
    The race diffusion simulation has two racers (a and b).
    '''
    np.random.seed(seed)
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
