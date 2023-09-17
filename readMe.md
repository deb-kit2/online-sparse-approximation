## Greedy Algorithms for Online Sparse Approximation


### Run for :
 - [x] HTP
 - [x] CoSaMP
 - [x] Iterative Hard Thresholding
 - [x] Subspace Pursuit : noisy

### Inplemented for :
 - [x] HTP
 - [x] CoSaMP
 - [x] Subspace Pursuit
 - [x] Iterative Hard Thresholding
 
### To-Do :
 - [ ] Re-run regret experiment with corrected IHT.
 - [ ] Be insightful about the variances of all random things.
 - [x] If doesn't converge, try with tau = 100 fixed.
 - [x] plots : reward vs time
 - [x] plots : regret vs time
 - [x] plots : average regret vs time
 - [x] plots : accumulated reward vs time
 - [x] make notebook.
 - [x] Other algorithms.
 - [x] Fix `x_best` and continue programming.

### Observations :
 - Converges with the log range of tau.

### Unanswered :
 - [x] Should I take `np.abs()` in the algorithms?
 
### Questions :
 - In regret calculation step, is `x_best` a constant through time?
 - `y_t` should be varying, not fixed. Try both.
 - Is `gamma` fixed?

### Answers
 - No. It changes with time. Obviously.
 - `y_t` will vary as `x_best` varies. I was wrong about the understanding. Now the setting asks me to try with `w_random` as fixed and varying.
 - Sir said, fixing it is valid for agnostic adversary scenario.