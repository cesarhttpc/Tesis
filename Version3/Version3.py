# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

np.random.seed(24)

# def dinamica(y,t,g,b):
#     x, v = y
#     dxdt = v
#     dvdt = g - b*v
#     return [dxdt, dvdt]

def dinamica(y,t,k,b):
    x, v = y
    dxdt = v
    dvdt = -k*x - b*v
    return [dxdt, dvdt]

'''
Parte 1: (Establecer la dinamica y simulacion de datos)

    Dado un par de parámetros g,b; se hará la "simulación de los datos" por medio de resolver la ecuacuón diferencial para dichos parámetros.

    # Declaración de los parámetros principales g,b.
    # Para resolver la ODE se require un conjunto de tiempos donde resolver, para ello se establece el vector t.
    # Usando el método dado por odeint, se requier condiciones iniciales para resolver la ODE (dinamica).

'''

# Parametros principales
g = 4.34
b = 1.15

# Simular los tiempos de observación
from scipy.stats import uniform
n = 31      # Tamaño de muestra
cota = 8
t = np.linspace(0,cota,num = n)

# ECUACIÓN DIFERENCIAL:
# Condiciones iniciales (posición, velocidad)
y0 = [1, 0.0]  

# Soluciones de la ecuación dínamica
solutions = odeint(dinamica, y0 ,t, args=(g,b))

# Coordenadas de caída amortiguada
x = solutions[:,0]
v = solutions[:,1]

# Añadir ruido a los datos
from scipy.stats import norm
error = norm.rvs(0,0.01,n)
error[0] = 0
x = x + error

# Grafica
plt.title('Datos simulados con k = %2.2f , b = %2.2f ' % (g, b))
plt.xlabel('Tiempo')
plt.ylabel('Posición')
plt.scatter(t,x, color= 'green')
plt.show()
# # %%

'''
Parte 2: (Establecimiento de la log-posterior)
'''

def logposterior(g, b, t, x, sigma = 1, alpha = 100, beta = 10, g_0 = 10, b_0 = 1):
   
    if g>0 and b>0:
        solution = odeint(dinamica, y0, t, args=(g,b))
        x_theta = solution[:,0]

        Logf_post = -n*np.log(sigma) - np.sum((x-x_theta)**2) /(2*sigma**2) + (alpha -1)*np.log(g) - alpha*g/g_0 + (beta - 1)*np.log(b) - beta*b/b_0

        return Logf_post
    else:
        Logf_post = -10**100
        return Logf_post 


'''
Parte 3: (Realizacion de la cadena por MCMC)

'''

def MetropolisHastingsRW(t_datos,x_datos,inicio, size = 50000 ,alpha =100, beta = 10, g_0 = 10, b_0 = 1 ):

    # Punto inicial (parametros)
    x = inicio

    sigma1, sigma2 = 0.3, 0.05

    # 
    sample = np.zeros([size,3])
    sample[0,0] = x[0]  
    sample[0,1] = x[1]
    sample[0,2] = logposterior(x[0], x[1], t_datos, x_datos, alpha=alpha, beta = beta, g_0 = g_0, b_0 = b_0)

    for k in range(size-1):

        # Simulacion de propuesta
        e1 = norm.rvs(0,sigma1)
        e2 = norm.rvs(0,sigma2)
        e = np.array([e1,e2])
        y = x + e   
        # ////////////////////////
        # //////////////////////// se reescribe x ????????
        # ////////////////////////

        # Cadena de Markov
        log_y = logposterior(y[0], y[1], t_datos, x_datos, alpha = alpha, beta = beta, g_0 = g_0, b_0 = b_0)
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

# Parametros de distribucion a prioi
g_0 = 5
alpha = 1
b_0 = 2
beta = 1.5

sample = MetropolisHastingsRW(t, x, inicio, size = 60000, g_0 = g_0, b_0 = b_0, beta = beta, alpha= alpha)

# %%
'''
Parte 4: (visualizacion)

'''

g_sample = sample[:,0]
b_sample = sample[:,1]
log_post = sample[:,2]

burn_in = 5000

plt.title('Cadena')
plt.plot(g_sample[:10000],label = 'k')
plt.plot(b_sample[:10000],label = 'b')
plt.legend()
plt.show()

plt.title('Trayectoria de caminata aleatoria')
plt.plot(g_sample,b_sample,linewidth = .5, color = 'gray')
plt.xlabel('k')
plt.ylabel('b')
plt.show() 

plt.title('LogPosterior de la cadena')
plt.plot(log_post, color = 'red')
plt.show()

from scipy.stats import gamma

plt.title('Distribuciones a priori y posterior para k')
plt.hist(g_sample[burn_in:], density= True, bins = 40)
dom_g = np.linspace(0,15,500)
plt.plot(dom_g, gamma.pdf(dom_g, a = alpha , scale = g_0/alpha))
plt.ylabel(r'$f(k)$')
plt.xlabel(r'$k$')
linea = np.linspace(0,0.5, 100)
linea_x = np.ones(100)
plt.plot(linea_x*g, linea, color = 'black', linewidth = 0.5)
plt.show()

plt.title('Distribuciones a priori y posterior para b')
plt.hist(b_sample[burn_in:], density= True, bins = 40)
dom_b = np.linspace(0,8,500)
plt.plot(dom_b, gamma.pdf(dom_b, a = beta, scale = b_0/beta))
plt.ylabel(r'$f(b)$')
plt.xlabel(r'$b$')
linea = np.linspace(0,1, 100)
linea_x = np.ones(100)
plt.plot(linea_x*b, linea, color = 'black', linewidth = 0.5)
plt.show()

estimador_g = np.mean(g_sample[burn_in:])    
estimador_b = np.mean(b_sample[burn_in:])
print('Estimador de k: ', estimador_g)  
print('Estimador de b: ', estimador_b)

# Grafica de la curva estimada
t_grafica = np.linspace(0,cota, 100)
solucion_estimada = odeint(dinamica, y0, t_grafica ,args=(estimador_g,estimador_b))

x_estimado = solucion_estimada[:,0]
v_estimado = solucion_estimada[:,1]

plt.title('Curva estimada (n = %u)'%(n-1))
plt.scatter(t,x, color= 'green', label = 'Datos sim. k = %2.2f, b = %2.2f' %(g,b))
plt.plot(t_grafica, x_estimado, label='Estimacion k = %2.2f, b = %2.2f' %(estimador_g, estimador_b), color = 'purple')
plt.xlabel('t')
plt.ylabel('Posición')
plt.legend()
plt.show()


# %%
# Grid
g_dom = np.linspace(0,20, num = 21)
b_dom = np.linspace(0,6,13)
Z = np.meshgrid(g_dom,b_dom)
plt.scatter(Z[0],Z[1])
plt.scatter(4.3, 3.3)
plt.ylabel(r'$b$')
plt.xlabel(r'$k$')


