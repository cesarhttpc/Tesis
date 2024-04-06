
import numpy as np
import matplotlib.pyplot as plt


def factorial(n):
    fact = 1
    for i in range(1, n+1):
        fact = fact * i
    return fact

def posterior(k,x,alpha,beta):

    n = len(x)
    suma = sum(x_datos)

    numerador = factorial(k)**n
    denominador = 1
    for i in range(0,n):
        denominador = denominador*factorial(k-x[i])

    arriba = factorial(k*n - suma + beta-1)
    abajo = factorial(k*n+alpha+beta-1)

    post = (numerador/denominador)*(arriba/abajo)
    return post


# Priori beta
alpha = 1
beta = 1

# Datos (observaciones)
x_datos = np.array([4,3,1,6,6,6,5,5,5,1])

# Posterior

K = np.arange(6,15)
m = len(K)
Z = np.zeros(m)
# print(Z)
for j in range(0,len(K)):
    Z[j] = posterior(K[j],x_datos,alpha,beta)


plt.title('Distribución posterior para k')
plt.xlabel('k')
plt.ylabel(r'$\pi(k|x^n)$')
plt.scatter(K,Z, color = 'orange')
plt.show()