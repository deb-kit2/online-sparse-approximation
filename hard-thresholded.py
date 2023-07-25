import numpy as np


def ft_hard_thresholded_pl(eta, mu, K, phi, T, x_best) :
    """
    Follow the Hard-Thresholded Perturbed Leader algorithm.

    Inputs :
    eta : multiplicative factor for perturbation
    mu : # to-do
    phi : shape(M, N)
    K : sparsity constant, number of non-zero elements
    T : timesteps to run the simulation for
    x_best : to be used for revealing next y

    Returns : 
    A dictionary containing `rewards`, `running_Y`, and `running_X`
    """

    # init
    M, N = phi.shape
    running_Y = []
    running_X = []
    rewards = []
    
    Y = np.zeros(M)
    gamma = np.random.normal(loc = 0.0, scale = 1.0, size = M) # constant
    
    for t in range(1, T + 1) :
        z = np.zeros(N)
        b = (Y + eta * gamma) / t
        
        tau = np.log(t) + 2
        for s in range(tau) :
            # abs?
            L = np.argsort(z + mu * phi.T @ (b - phi @ z))[::-1][:K]
            z_new = np.linalg.lstsq(phi[:, L], b, rcond = None)[0]
            z[L] = z_new

        x = z
        # to-do : reveal y
        y = phi @ x_best + w_random

        # get reward
        reward = np.dot(y, phi @ x) - 0.5 * np.linalg.norm(phi @ x) ** 2
        Y = Y + y

        rewards.append(reward)
        running_X.append(x)
        running_Y.append(y)

    return {
        "rewards" : rewards,
        "running_X" : running_X,
        "running_Y" : running_Y
    }


if __name__ == "__main__" :
    # to-do 
    pass
