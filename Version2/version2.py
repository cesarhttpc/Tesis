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
g = 9.81    
b = 3.9

# Simular los tiempos de observación
from scipy.stats import uniform
n = 50      # Tamaño de muestra
t = uniform.rvs(0, 1, n)       
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
x = x + norm.rvs(0,0.05,n)

# Grafica
plt.title('Datos simulados con g = %u , b = %u ' % (g, b))
plt.xlabel('Tiempo')
plt.ylabel('Posición')
plt.scatter(t,x, color= 'green')

# # %%

'''
Parte 2: (Establecimiento de la log-posterior)
'''

def logposterior(g, b, t, x, sigma = 1, gamma = 100, beta = 10, g_0 = 10, b_0 = 1):
   
    if g>0 and b>0:
        solution = odeint(dinamica, y0, t, args=(g,b))
        x_theta = solution[:,0]

        Logf_post = -n*np.log(sigma) - np.sum(x-x_theta)**2 /(2*sigma**2) + (gamma -1)*np.log(g) - gamma*g/g_0 + (beta - 1)*np.log(b) - beta*b/b_0

        return Logf_post
    else:
        Logf_post = -10**100
        return Logf_post 


'''
Parte 3: (Realizacion de la cadena por MCMC)

'''

def MetropolisHastingsRW(t_datos,x_datos,inicio, size = 50000 ):

    # Punto inicial (parametros)
    x = inicio

    sigma1, sigma2 = 0.3, 0.05

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

inicio = np.array([8,3])

sample = MetropolisHastingsRW(t, x, inicio)

# %%

'''
Parte 4: (visualizacion)

'''
g_sample = sample[:,0]
b_sample = sample[:,1]
log_post = sample[:,2]

burn_in = 5000
plt.title('Cadena')
plt.plot(g_sample[:10000],label = 'g')
plt.plot(b_sample[:10000],label = 'b')
plt.legend()
plt.show()

# plt.plot(g_sample)
# plt.show()

# plt.plot(b_sample)
# plt.show()

plt.title('Trayectoria de caminata aleatoria')
plt.plot(g_sample,b_sample,linewidth = .5, color = 'gray')
plt.xlabel('g')
plt.ylabel('b')
plt.show() 

plt.title('LogPosterior de la cadena')
plt.plot(log_post, color = 'red')
plt.show()

from scipy.stats import gamma

plt.title('Distribuciones a priori y posterior para g')
plt.hist(g_sample[burn_in:], density= True, bins = 30)
alpha = 100
dom_g = np.linspace(7,17,500)
plt.plot(dom_g, gamma.pdf(dom_g, a = alpha , scale = 10/alpha))
plt.ylabel(r'$f(g)$')
plt.xlabel(r'$g$')
plt.show()

plt.title('Distribuciones a priori y posterior para b')
plt.hist(b_sample[burn_in:], density= True, bins = 30)
beta = 10
dom_b = np.linspace(0,4,500)
plt.plot(dom_b, gamma.pdf(dom_b, a = beta, scale = 1/beta))
plt.ylabel(r'$f(b)$')
plt.xlabel(r'$b$')
plt.show()

estimador_g = np.mean(g_sample[burn_in:])    
estimador_b = np.mean(b_sample[burn_in:])
print('Estimador de g: ', estimador_g)  
print('Estimador de b: ', estimador_b)

t_grafica = np.linspace(0,1, 100)
solucion_estimada = odeint(dinamica, y0, t_grafica ,args=(estimador_g,estimador_b))

x_estimado = solucion_estimada[:,0]
v_estimado = solucion_estimada[:,1]

plt.plot(t_grafica, x_estimado, label='Solucion Estimada')
plt.scatter(t,x, color= 'green')





