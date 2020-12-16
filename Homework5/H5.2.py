import numpy as np

a = 2
s = 6
gamma = 0.6
N = 10

Q = np.zeros([a,s])

R = np.zeros([s,s])
R[0,0]=0
for i in range(s):
    for j in range(s):
        if i==0 and j==0: continue
        if i==j: R[i,j]=(i+4)**(-.5)
        else: R[i,j]=np.absolute(j-i)**(1/3)
#print(R)

T = np.zeros([a, s, s])
T[:,0,0]=1 #state 0 is terminal
for i in range(1,4):
    T[1,i,i-1]=1
    T[0,i,i+2]=0.7
    T[0,i,i]=0.3
for i in range(4,6):
    T[1,i,i-1]=1
    T[0,i,i]=1
#print(T)

#for i in range(N):
Q = np.sum(T*(R+gamma*np.max(Q[:,:,None], axis=0)), axis=2)
print("Q=",Q)

V = np.max(Q,axis=0)
print("V=",V)

pi = np.argmax(Q, axis=0)
print("pi{C,M}=",pi)