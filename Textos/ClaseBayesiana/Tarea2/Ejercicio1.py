# %%
import numpy as np
from scipy.stats import pareto
import matplotlib.pyplot as plt

alpha = 5
beta = 1

# Variables Pareto
muestra = pareto.rvs(b = alpha, scale = beta, size = 50000)

# PDF Pareto
funcion = lambda x: alpha*(beta**alpha) / x**(alpha + 1)

# Plots Pareto
t = np.linspace(beta, 4,100)
plt.title(r'Distribución Pareto($\alpha = %r , \beta = %r$)'%(alpha,beta))
plt.plot(t,funcion(t))
plt.hist(muestra,bins = 100, density= True, alpha = 0.6)
plt.show()

# %%
# A priori
datos= np.array([7.7, 1.2, 5.5, 5.5, 13.0, 12.0, 9.7, 11.0, 18.0, 19.0, 16.0, 9.3, 6.8, 3.7, 4.8, 6.1, 2.2, 3.6, 9.9, 8.0, 19.0, 11.0, 15.0, 8.1, 15.0, 5.1, 12.0, 7.3, 2.1, 14.0])

alpha_0 = 5
beta_0 = 1

alpha = alpha_0 + len(datos)
beta = max(beta_0, max(datos))
print(beta)

plt.title(r'Distribución a priori $f(\theta)$')
t_0 = np.linspace(beta_0,1.2,100)
plt.plot(t_0,funcion(t_0), label = r'$\alpha_0 = $ %r , $\beta_0 = $ %r' %(alpha_0,beta_0), color = 'green')
plt.legend()
plt.show()

plt.title(r'Distribución posterior $f(\theta|X^n)$')
t = np.linspace(beta, 23,100)
plt.plot(t,funcion(t), label = r'$\alpha_0 = $ %r , $\beta_0 = $ %r' %(alpha,beta))
plt.legend()
plt.show()


fig, axs = plt.subplots(1,2,figsize = (10,4))
fig.suptitle(r'Distribuciones a priori $f(\theta)$ y posterior $f(\theta|X^n) $ para $\theta$')
axs[0].plot(t_0,funcion(t_0), color = 'green')
axs[1].plot(t,funcion(t))
# axs[0].label_outer()
# axs[1].label_outer()
# plt.savefig('Figures/distribuciones.png')
plt.show()

plt.title(r'Histograma de la muestra $X_i\sim U(0,\theta)$')
plt.hist(datos, bins= 21)
plt.show()

