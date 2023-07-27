import numpy as np
from tqdm import tqdm


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
        
        y_ = np.zeros(M)
        x_best_ += x_b
        offline_loss = 0
        for j in range(i + 1) :
            y_ += running_Y[j]
            offline_loss += 0.5 * np.linalg.norm(y_ - phi @ x_best_)

        regrets.append(online_loss - offline_loss)
        average_regrets.append((online_loss - offline_loss) / (i+1))

    history["regrets"] = regrets
    history["average_regrets"] = average_regrets

    return history
