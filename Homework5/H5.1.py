import numpy as np

gamma = 0.75

R = np.array([[0,1,1,10],[1,1,10,0]])
V_0 = np.array([0,0,0,0])

a = 2
s = 4

T = np.zeros([a, s, s])
for i in range(s):
    if i>0: T[0,i,i-1]=1
    if i<s-1: T[1,i,i+1]=1

#print(T)

V = V_0
V_sp = np.zeros([a,s])
V_sp[0,1:s] = V[:s-1]
V_sp[1,:s-1] = V[1:s]
for i in range(100):
    V_sp[0,1:s] = V[:s-1]
    V_sp[1,:s-1] = V[1:s]
    V = np.max(np.sum(T*(R + gamma*V_sp)[:,:,None], axis=2), axis=0)
    print("V= ",V,"after i=",i+1)

