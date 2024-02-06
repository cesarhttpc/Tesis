# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# %%

def libre(y,t,g,b):
    x, v = y
    dxdt = v
    dvdt = g
    return [dxdt, dvdt]

def damping(y,t,g,b):
    x, v = y
    dxdt = v
    dvdt = g - b*v
    return [dxdt, dvdt]

def resorte(y,t,k,b):
    x, v = y
    dxdt = v
    dvdt = -k*x - b*v
    return [dxdt, dvdt]

def forward(g,b):
    
    solutions = odeint(damping, y0, t, args=(g,b))
    return solutions[:,0]

def logposterior(g, b, t, y_obs, t_obs, sigma = 1):
    
    # Parametros solución x
    n = len(y_obs)
    h = derecha/pasos    

    # Solución posición
    solucion = odeint(damping, y0, t, args=(g,b))
    x = solucion[:,0]

    # Parametro Priori 
    alpha = 1000

    # Log verosimilitud
    cuadrados = np.zeros(n)
    for j in range(n):

        cuadrados[j] = (y_obs[j]- x[int(t_obs[j]/h)])**2

        Logf_post = -n*np.log(sigma) - sum(cuadrados)/(2*sigma**2) + (alpha -1)*np.log(g) + 9*np.log(b) - alpha*g/10 - 10 * b

    return Logf_post


# Parametetros
g = 10
b = .7

# Condiciones iniciales
y0 = [0, 0.0]  

#Dominio solución
derecha = 1
pasos = 100
t = np.linspace(0, derecha, pasos)  

# Observaciones
y_obs = np.array([0, 0.06 ,0.16, 0.26, 0.36, 0.46, 0.56])
t_obs = np.array([0, 0.071, 0.158, 0.220, 0.267, 0.306, 0.340])









print(logposterior(g,b,t,y_obs, t_obs,))








# %%

# /////////// Visualización ////////


# Soluciones de la ecuación dínamica
solutions = odeint(damping, y0, t, args=(g,b))

solutions0 = odeint(libre, y0, t, args=(g,b))
k = 1
solutions1 = odeint(resorte,y0, t, args=(k,b))


# coordenadas de caída amortiguada
x = solutions[:, 0]
v = solutions[:, 1]


# Graficas
plt.plot(t, x, label='Caida amortiguada')
# plt.plot(t, v, label='Velocidad v')
plt.xlabel('Tiempo')
plt.ylabel('Posición')
plt.title('Mecánica Clasica')
plt.plot(t, solutions0[:,0], label = 'Caída libre')
plt.legend()
# plt.show()

# z = lambda x: 9.7*x**2/2
# plt.plot(t,z(t))
 
plt.plot( t_o, y_o, 'o')
plt.show()

plt.plot(x,v)
plt.title('Diagrama de fase')
plt.xlabel('Pocisión')
plt.ylabel('Velocidad')
plt.legend()
plt.show()






 



























# %%
from scipy.stats import gamma , norm

# Distribuciones a priori
dom = np.linspace(0,4,500)
dom1 = np.linspace(0,15,500)

plt.plot(dom, gamma.pdf(dom,a = 10, scale = .1))  # 5, 0.1
plt.title('priori para b ')
plt.show()

alpha = 1000
plt.plot(dom1, gamma.pdf(dom1, a = alpha , scale = 9.81/alpha))
plt.show()

plt.plot(dom1, norm.pdf(dom1, 10, 0.3))
plt.title('Priori para g')
plt.show()


# %%














