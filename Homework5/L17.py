import numpy as np

s = 5
a = 3
gamma = 0.5

R = np.zeros([s,s])
R[:,s-1]=1

Q_0 = np.zeros([a,s])
V_0 = np.array([0,0,0,0,0])


T = np.zeros([a, s, s])

for i in range(s):  # from i+1 to s-1 to remove extremities
    # decide to move left
    if i>0: T[0, i, i-1] = 1/3
    T[0, i, i] = 2/3

    # decide to stay
    if i>0: T[1, i, i-1] = 0.25
    T[1, i, i] = 0.5
    if i<s-1: T[1, i, i+1] = 0.25

    # decide to move right
    T[2, i, i] = 2/3
    if i<s-1: T[2, i, i+1] = 1/3

T[:2, 0, :2] = 0.5*np.ones([1, a-1, 2])  # leftmost s to s'
T[1:, s-1, s-2:] = 0.5*np.ones([1, a-1, 2])  # rightmost s to s'

#V = V_0
Q = Q_0
N = 100
for i in range(N):
    Q = np.sum(T*(R+gamma*np.max(Q[:,:,None], axis=0)), axis=2)
    #V = np.max(np.sum(T*(R + gamma*V_sp)[:,:,None], axis=2), axis=0)
print("V_100 = ", np.max(Q,axis=0))


Q = Q_0
#V = V_0
N = 200
for i in range(N):
    Q = np.sum(T*(R+gamma*np.max(Q[:,:,None], axis=0)), axis=2)
    #V = np.max(np.sum(T*(R + gamma*V), axis=2), axis=0)
print("V_200 = ", np.max(Q,axis=0))

Q = Q_0
#V = V_0
N = 10
for i in range(N):
    Q = np.sum(T*(R+gamma*np.max(Q[:,:,None], axis=0)), axis=2)
    #V = np.max(np.sum(T*(R + gamma*V), axis=2), axis=0)
print("V_10 = ", np.max(Q,axis=0))