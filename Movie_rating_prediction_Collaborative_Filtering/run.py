import sys
import numpy as np
import math
'''
ratings=open("/home2/e0268-26/a1/ratings.train","r")
N=ratings.readline()
a=int(N)
user=[]
movie=[]
rating=[]
user_test=[]
movie_test=[]
li=[]
for i in range(1,a+1):
    e = ratings.readline()
    li=list(e.split())
    user.append(int(li[0]))
    movie.append(int(li[1]))
    rating.append(float(li[2]))
num_user=max(user)
num_movie=max(movie)
num_features=100

Rate=np.zeros(shape=(num_user+1,num_movie+1))
for i in range(a):
    Rate[user[i]][movie[i]]=rating[i]
did_rate=(Rate!=0)*1.0
# M
X_ = np.random.rand(num_movie + 1, num_features)
# N
theta = np.random.rand(num_user + 1, num_features)

#www.quuxlabs.com/matrix-factorization
def matrix_factorization(R, P, Q, K, iter=50, alpha=0.003, reg_param=0.02):
    Q = Q.T
    for step in range(iter):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    diff = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * diff * Q[k][j] - reg_param * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * diff * P[i][k] - reg_param * Q[k][j])
    return np.dot(P,Q)

M = matrix_factorization(Rate,theta,X_,num_features)

np.round(M,6)
A= np.asarray(M)
A=np.clip(A,a_min=0.0,a_max=5.0)

np.savetxt("mat_.txt", A , fmt="%f")

'''
f=open("/home2/e0268-26/a1/mat_.txt","r")
f1=f.readlines()
e=[]
for i in f1:
    e.append(list(map(float,i.split())))

li2=[]
input=open(sys.argv[1],"r")
output=open(sys.argv[2],"w")
I=int(input.readline())
for i in range(I):
    k=input.readline()
    li2=list(k.split())
    output.write(str(e[int(li2[0])][int(li2[1])]))
    output.write("\n")
input.close()
output.close()
#ratings.close()


