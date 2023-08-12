import copy
import numpy as np
from tqdm import tqdm

def ft_cosamp_pl(eta, phi, K, T) :
    """
    Follow the CoSaMPed Perturbed Leader algorithm.

    Inputs :
    eta : multiplicative factor for perturbation
    phi : shape(M, N)
    K : sparsity constant, number of non-zero elements
    T : timesteps to run the simulation for

    Returns : 
    A dictionary containing `rewards`, `running_Y`, and `running_X`
    """

    # init
    M, N = phi.shape
    running_Y = []
    running_X = []
    running_X_best = []
    rewards = []
    support_x = np.random.permutation(N)[:K] # fixed support setting
    
    Y = np.zeros(M)
    gamma = np.random.normal(loc = 0.0, scale = 1.0, size = M) # constant
    
    for t in tqdm(range(1, T + 1)) :
        z = np.zeros(N)
        b = (Y + eta * gamma) / t
        
        # get x from oracle
        tau = int(np.log(t) + 10)
        for s in range(tau) :
            # abs? I think needed.
            sup1 = np.nonzero(z)
            sup2 = np.argsort(np.abs(phi.T @ (b - phi @ z)))[::-1][:2*K]

            L = np.union1d(sup1, sup2)
            u = np.linalg.lstsq(phi[:, L], b, rcond = None)[0]
            
            topK = np.argsort(np.abs(u))[::-1][:K]
            z = np.zeros(N)
            z[topK] = u[topK]

        x = z

        # reveal y
        x_best = np.zeros(N)
        x_best[support_x] = np.random.normal(loc = 0.0, scale = 1.0, size = K)
        noise = np.random.normal(scale = 1/128, size = M)

        y = (phi @ x_best) + noise

        # get reward
        reward = np.dot(y, phi @ x) - 0.5 * np.linalg.norm(phi @ x) ** 2
        Y = Y + y

        rewards.append(reward)
        running_X.append(x)
        running_Y.append(y)
        running_X_best.append(x_best)

    accumulated_rewards = copy.deepcopy(rewards)
    for i in range(1, len(accumulated_rewards)) :
        accumulated_rewards[i] += accumulated_rewards[i-1]

    for i in range(len(accumulated_rewards)) :
        accumulated_rewards[i] /= i+1

    return {
        "phi" : phi,
        "rewards" : rewards,
        "average_rewards" : accumulated_rewards,
        "running_X" : running_X,
        "running_X_best" : running_X_best,
        "running_Y" : running_Y
    }
