import copy
import numpy as np
import tqdm


def hard_threshold(vector, sparsity : int) :
    """
    Performs hard thresholding of the given `vector` with `sparsity`.
    """

    result = np.zeros_like(vector)
    indices = np.argsort(np.abs(vector))[::-1][:sparsity]

    result[indices] = vector[indices]
    
    return result


def htp(phi, y, K, mu = 0.3, max_iter = 100) :
    """
    Performs Hard-Thresholded Perturbing

    Inputs :
    phi : shape(M, N)
    y : shape(M)
    K : sparsity level
    mu : multiplicative factor for HTP,
    max_iter : number of steps before convergence
    """
    tol = 1e-6

    M, N = phi.shape
    x = np.zeros(N)

    for t in range(max_iter) :
        L = np.argsort(np.abs(x + mu * phi.T @ (y - phi @ x)))[::-1][:K]
        
        x_new = np.zeros(N)
        x_new[L] = np.linalg.lstsq(phi[:, L], y, rconf = None)[0]

        if np.linalg.norm(x_new - x) < tol * np.linalg.norm(x) :
            break

        x = x_new

    return x


def ihtp(phi, y, K, max_iter = 100) :
    """
    Performs Iterative Hard-Thresholded Perturbing

    Inputs :
    phi : shape(M, N)
    y : shape(M)
    K : sparsity level
    max_iter : number of steps before convergence
    """
    tol = 1e-6

    M, N = phi.shape
    x = np.zeros(N)

    for t in range(max_iter) :
        r = x + phi.T @ (y - phi @ x)
        L = np.argsort(np.abs(r))[::-1][:K]
        
        x_new = np.zeros(N)
        x_new[L] = r[L]

        if np.linalg.norm(x_new - x) < tol * np.linalg.norm(x) :
            break
        
        x = x_new

    return x


def cosamp(phi, y, K, max_iter = 100) :
    """
    Performs CoSaMPed Perturbing

    Inputs :
    phi : shape(M, N)
    y : shape(M)
    K : sparsity level
    max_iter : number of steps before convergence
    """
    tol = 1e-6

    M, N = phi.shape
    x = np.zeros(N)

    for t in range(max_iter) :
        sup1 = np.nonzero(x)
        sup2 = np.argsort(np.abs(phi.T @ (y - phi @ x)))[::-1][:2*K]

        L = np.union1d(sup1, sup2)
        u = np.linalg.lstsq(phi[:, L], y, rcond = None)[0]

        topK = np.argsort(np.abs(u))[::-1][:K]
        x_new = np.zeros(N)
        x_new[topK] = u[topK]

        if np.linalg.norm(x_new - x) < tol * np.linalg.norm(x) :
            break
        
        x = x_new

    return x


def residual(y, A) :
    yp = (A @ np.linalg.pinv(A)) @ y
    res = y - yp
    return res


def sp(phi, y, K, max_iter = 25) :
    """
    Performs Subspace Pursuit

    Inputs :
    phi : shape(M, N)
    y : shape(M)
    K : sparsity level
    max_iter : number of steps before convergence
    """

    M, N = phi.shape

    tl = np.argsort(np.abs(phi.T @ y))[::-1][:K]
    r = residual(y, phi[:, tl])

    for t in range(max_iter) :
        sup = np.argsort(np.abs(phi.T @ r))[::-1][:K]
        tl_ = np.union1d(tl, sup)

        x_p = np.linalg.pinv(phi[:, tl_]) @ y

        tl_new = np.argsort(np.abs(x_p))[::-1][:K]

        if tl == tl_new :
            break
        
        r = residual(y, phi[:, tl_new])

    x = np.zeros(N)
    x[tl] = np.linalg.pinv(phi[:, tl]) @ y

    return x


def solver(max_support_size, support_size, phi, T) :
    """
    A function to test some hypothesis.

    Inputs :
    max_support_size : 
    support_size : 
    phi : shape(M, N)
    T : time-steps to run the simulation for
    """

    M, N = phi.shape
    support = np.random.permutation(N)[:max_support_size]
    
    errors = {
        "htp" : [],
        "ihtp" : [],
        "cosamp" : [],
        "subspace" : []
        }
    Y = np.zeros(M)
    U = np.zeors(N)

    for t in tqdm(range(1, T + 1)) :
        # u_t
        u_t = np.zeros(N)
        support_t = np.random.choice(support, size = support_size, replace = False)
        u_t[support_t] = np.random.normal(scale = 1.0, size = support_size) #

        # noise
        w_t = np.random.normal(scale = 1.0, size = M)

        y_t = phi @ u_t + w_t
        
        Y += y_t
        U += u_t

        h = htp(phi, Y / t, support_size)
        ih = ihtp(phi, Y / t, support_size)
        c = cosamp(phi, Y / t, support_size)
        s = sp(phi, Y / t, support_size)

        expected = hard_threshold(U / t, support_size)

        errors["htp"].append(np.mean((expected - h) ** 2))
        errors["ihtp"].append(np.mean((expected - ih) ** 2))
        errors["cosamp"].append(np.mean((expected - c) ** 2))
        errors["sp"].append(np.mean((expected - s) ** 2))

    return errors
