import numpy as np
import em
import naive_em
import common


X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0

mixture, post = common.init(X, K, seed)


#post, LL = naive_em.estep(X, mixture)
post, LL = em.estep(X, mixture)

print("After first E-step:")
print("post:")
print(post[0:3,:])
print("LL:", LL)

#mixture = naive_em.mstep(X, post)
mixture = em.mstep(X, post, mixture)
print()
print("After first M-step:")
print("Mu:")
print(mixture.mu)
print("Var:", mixture.var)
print("P:", mixture.p)

mixture, post, LL = em.run(X, mixture, post)
print()
print("After a run:")
print("Mu:")
print(mixture.mu)
print("Var:", mixture.var)
print("P:", mixture.p)
print("post:")
print(post)
print("LL:", LL)

