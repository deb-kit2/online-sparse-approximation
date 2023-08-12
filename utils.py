import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns


def calculate_regret(history : dict) :
    
    running_X = history["running_X"]
    running_Y = history["running_Y"]
    phi = history["phi"]
    running_X_best = history["running_X_best"]

    M, N = phi.shape
    regrets = []
    average_regrets = []
    
    online_loss = 0
    x_best_ = np.zeros(N)
    for i, (x, y, x_b) in tqdm(
        enumerate(zip(running_X, running_Y, running_X_best)),
        total = len(running_X)
        ) :
        online_loss += 0.5 * np.linalg.norm(y - phi @ x) ** 2 
        
        x_best_ += x_b
        offline_loss = 0
        for j in range(i + 1) :
            offline_loss += 0.5 * np.linalg.norm(running_Y[j] - phi @ (x_best_ / (i+1)))

        regrets.append(online_loss - offline_loss)
        average_regrets.append((online_loss - offline_loss) / (i+1))

    history["regrets"] = regrets
    history["average_regrets"] = average_regrets

    return history


def save(M, N, K, T, eta, mu, df, scale, alg = "htp") :
    
    fig, ax = plt.subplots(2, 2, figsize = (8, 5), dpi = 200)

    a = sns.lineplot(df[["Regret"]][:200], ax = ax[0][0])
    b = sns.lineplot(df[["Reward"]][:200], ax = ax[0][1])
    c = sns.lineplot(df[["Average Regret"]][:200], ax = ax[1][0])
    f = sns.lineplot(df[["Average Reward"]][:200], ax = ax[1][1])

    plt.savefig(f"logs/{alg}_M{M}_N{N}_K{K}_T{T}_eta{eta}_mu{mu}_scale{scale}.png")

    plt.close()
    