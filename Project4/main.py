import numpy as np
import kmeans
import common
import naive_em
import em

#X = np.loadtxt("toy_data.txt")
X = np.loadtxt("netflix_incomplete.txt")
#X = X[~np.all(X==0, axis=1)]
"""
print("K-means")
for K in np.arange(4):
    cost_min = float('inf')
    for seed in np.arange(5):
        mixture, post = common.init(X, K+1, seed)
        mixture, post, cost = kmeans.run(X, mixture, post)
        #common.plot(X, mixture, post, "K-means, K="+str(K)+" seed="+str(seed))
        cost_min = np.min([cost_min, cost])

    print("K =", K+1, " cost =", cost_min)

print()
print("E-M")
"""
best_K = None
best_bic = float('-inf')
for K in [0, 11]:
    ll_max = float('-inf')
    best_seed = None
    best_mixture = None
    for seed in range(5):
        mixture, post = common.init(X, K+1, seed)
        mixture, post, ll = em.run(X, mixture, post)
        full_matrix = em.fill_matrix(X, mixture)
        #common.plot(X, mixture, post, "E-M, K="+str(K)+" seed="+str(seed))
        if ll > ll_max:
            best_seed = seed
            ll_max = ll
            best_mixture =mixture
    """
    bic = common.bic(X, best_mixture, ll_max)
    if bic > best_bic:
        best_K = K+1
        best_bic = bic
    """
    print("K =", K+1, " LL =", ll)
#print("full_matrix =")
#print(full_matrix[4,:])
#print("Best K=", best_K, " Best BIC=", best_bic)
