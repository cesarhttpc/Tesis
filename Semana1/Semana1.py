# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import uniform, norm, gamma


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

def MetropolisHastingsRW(tamañoMuestra = 20000, propuesta = 'Normal'):

    # Punto inicial (parametros)
    x = np.array([30,9])

    Muestra1 = np.zeros(tamañoMuestra)  # g
    Muestra2 = np.zeros(tamañoMuestra)  # b
    # Muestra3 = np.zeros(tamañoMuestra)  # sigma
    Muestra1[0] = x[0]
    Muestra2[0] = x[1] 
    # Muestra3[0] = x[2]


    for k in range(tamañoMuestra-1):

        # Simulación de propuesta
        if propuesta == 'Normal':

            sigma1 = 0.5
            sigma2 = 0.5
            e1 = norm.rvs(0,sigma1)
            e2 = norm.rvs(0,sigma2)
            e = np.array([e1,e2])
            y = x + e 

        # if propuesta == 'SigmaDesconocido':
        #     sigma1 = 0.5
        #     sigma2 = 0.05
        #     sigma3 = 0.5

        #     e1 = norm.rvs(0,sigma1)
        #     e2 = norm.rvs(0,sigma2)
        #     e3 = norm.rvs(0,sigma3)

        #     e = np.array([e1,e2,e3])
        #     y = x + e 


        # Cadena de Markov
        cociente = np.exp(logposterior(y[0],y[1],t,y_obs,t_obs)- logposterior(x[0],x[1],t,y_obs,t_obs))

        # Transición de la cadena
        if uniform.rvs(0,1) < cociente :    #Ensayo Bernoulli
            Muestra1[k+1] = y[0]
            Muestra2[k+1] = y[1]
            x = y
        else:
            Muestra1[k+1] = x[0]
            Muestra2[k+1] = x[1]
        
    return Muestra1,Muestra2



# %%

# Condiciones iniciales
y0 = [0, 0.0]  

#Dominio solución
derecha = 0.5
pasos = 100
t = np.linspace(0, derecha, pasos)  

# Observaciones
y_obs = np.array([0, 0.06 ,0.16, 0.26, 0.36, 0.46, 0.56])
t_obs = np.array([0, 0.071, 0.158, 0.220, 0.267, 0.306, 0.340])

# t_obs2 = np.array([0, 0.072, 0.158, 0.219, 0.265, 0.304, 0.338])
# t_obs3 = np.array([0, 0.078, 0.163, 0.224, 0.270, 0.309, 0.343])




Posterior_g, Posterior_b = MetropolisHastingsRW()

# %%

plt.title('Cadena')
plt.plot(Posterior_g[:5000])
plt.plot(Posterior_b[:5000])
plt.show()

plt.title('Trayectoria de caminata aleatoria')
plt.plot(Posterior_g,Posterior_b,linewidth = .5, color = 'gray')
plt.xlabel('g')
plt.ylabel('b')
plt.show()  

plt.title('LogPosterior de la cadena')
size = len(Posterior_g)
z = np.zeros(size)
enteros = np.arange(size)
for g,b,k in zip(Posterior_g,Posterior_b,enteros):

    # for k in range(size):

    z[k] = -logposterior(g,b,t,y_obs,t_obs)

plt.plot(z)
plt.show()

burn_in = 1000

plt.title('Distribución posterior de g')
plt.hist(Posterior_g[burn_in:], bins = 30)
plt.show()

plt.title('Distribución posterior de b')
plt.hist(Posterior_b[burn_in:], bins = 30)
plt.show()

estimador_g = np.mean(Posterior_g[burn_in:])    
estimador_b = np.mean(Posterior_b[burn_in:])
print('Estimador de g: ', estimador_g)  
print('Estimador de b: ', estimador_b)

# %%
# /////////// Visualización //////////

# Parametetros
g = estimador_g
b = estimador_b

# Soluciones de la ecuación dínamica
solutions = odeint(damping, y0, t, args=(g,b))

solutions0 = odeint(libre, y0, t, args=(g,b))
k = 1
solutions1 = odeint(resorte,y0, t, args=(k,b))


# coordenadas de caída amortiguada
x = solutions[:, 0]
v = solutions[:, 1]


# Graficas
plt.title('Dinamica sistema')
plt.plot(t, x, label='Caida amortiguada')
plt.plot(t, solutions0[:,0], label = 'Caída libre')
plt.plot( t_obs, y_obs, 'o')
# plt.plot(t, v, label='Velocidad v')
plt.xlabel('Tiempo')
plt.ylabel('Posición')
plt.legend() 
plt.show()

plt.plot(x,v)
plt.title('Diagrama de fase')
plt.xlabel('Pocisión')
plt.ylabel('Velocidad')
plt.legend()
plt.show()

# Distribuciones a priori


dom_g = np.linspace(0,15,500)
dom_b = np.linspace(0,4,500)

alpha = 1000
plt.plot(dom_g, gamma.pdf(dom_g, a = alpha , scale = 10/alpha))
plt.title('Distribución a priori de g')
plt.ylabel(r'$f(g)$')
plt.xlabel(r'$g$')
plt.show()

plt.plot(dom_b, gamma.pdf(dom_b, a = 10, scale = .1))  # 5, 0.1
plt.title('Distribución a priori de b ')
plt.ylabel(r'$f(b)$')
plt.xlabel(r'$b$')
plt.show()

# %%














