# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

np.random.seed(24)

def dinamica(y,t,g,b):
    x, v = y
    dxdt = v
    dvdt = g - b*v
    return [dxdt, dvdt]

'''
Parte 1: (Establecer la dinamica y simulacion de datos)

    Dado un par de parámetros g,b; se hará la "simulación de los datos" por medio de resolver la ecuacuón diferencial para dichos parámetros.

    # Declaración de los parámetros principales g,b.
    # Para resolver la ODE se require un conjunto de tiempos donde resolver, para ello se establece el vector t.
    # Usando el método dado por odeint, se requier condiciones iniciales para resolver la ODE (dinamica).

'''

# Parametros principales
g = 10
b = 1

# Simular los tiempos de observación
from scipy.stats import uniform
n = 20      # Tamaño de muestra
t = uniform.rvs(0, 5, n)       
t = np.sort(t)

# ECUACIÓN DIFERENCIAL:
# Condiciones iniciales (posición, velocidad)
y0 = [0, 0.0]  

# Soluciones de la ecuación dínamica
solutions = odeint(dinamica, y0 ,t, args=(g,b))

# Coordenadas de caída amortiguada
x = solutions[:,0]
v = solutions[:,1]

# Añadir ruido a los datos
from scipy.stats import norm
x = x + norm.rvs(0,0.2,n)

# Grafica
plt.title('Datos simulados con g = %u , b = %u ' % (g, b))
plt.xlabel('Tiempo')
plt.ylabel('Posición')
plt.scatter(t,x, color= 'orange')

# # %%

'''
Parte 2: (Establecimiento de la log-posterior)
'''

def logposterior(g, b, t, x, sigma = 1, gamma = 100, beta = 10, g_0 = 10, b_0 = 1):
   
    solution = odeint(dinamica, y0, t, args=(g,b))
    x_theta = solution[:,0]

    Logf_post = -n*np.log(sigma) - np.sum(x-x_theta)**2 /(2*sigma**2) + (gamma -1)*np.log(g) - gamma*g/g_0 + (beta - 1)*np.log(b) - beta*b/b_0

    return Logf_post



# # %%
'''
Parte 3: ()

'''

def MetropolisHastingsRW(t_datos,x_datos,inicio, size = 10000 ):

    # Punto inicial (parametros)
    x = inicio

    sigma1, sigma2 = 1, 1

    # 
    sample = np.zeros([size,3])
    sample[0,0] = x[0]  
    sample[0,1] = x[1]
    sample[0,2] = logposterior(x[0], x[1], t_datos, x_datos)

    for k in range(size-1):

        # Simulacion de propuesta
        e1 = norm.rvs(0,sigma1)
        e2 = norm.rvs(0,sigma2)
        e = np.array([e1,e2])
        y = x + e 

        # Cadena de Markov
        log_y = logposterior(y[0], y[1], t_datos, x_datos)
        # log_x = logposterior(x[0], x[1], t_datos, x_datos)
        log_x = sample[k,2] # Recicla logverosimilitud
        cociente = np.exp( log_y - log_x )

        # Transicion de la cadena
        if uniform.rvs(0,1) <= cociente:

            sample[k+1,0] = y[0]
            sample[k+1,1] = y[1] 
            sample[k+1,2] = log_y

            x = y
        else:
            
            sample[k+1,0] = x[0]
            sample[k+1,1] = x[1] 
            sample[k+1,2] = log_x

    return sample

        

inicio = np.array([14,5])

# MetropolisHastingsRW(t,x, inicio)

sample = MetropolisHastingsRW(t,x, inicio)

# %%
g_sample = sample[:,0]
b_sample = sample[:,1]
log_post = sample[:,2]


plt.plot(g_sample)
plt.show()

plt.plot(b_sample)
plt.show()

plt.plot(g_sample,b_sample)
plt.show()

plt.plot(log_post)

plt.hist(g_sample, density= True, bins = 20)
plt.show()
plt.hist(b_sample, density= True, bins = 20)
