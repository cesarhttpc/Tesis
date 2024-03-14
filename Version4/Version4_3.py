# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.spatial import cKDTree
from scipy.stats import gamma, uniform, norm
import time

class VecinosCercanos:

    def __init__(self,puntos):
        # Construir el árbol cKDTree con los puntos
        self.arbol = cKDTree(puntos)
        self.solutions = {}  # Dictionary to store precomputed solutions

    def compute_solutions(self, t, puntos_malla):
        for i, punto in enumerate(puntos_malla):
            solution = odeint(dinamica, y0, t, args=(punto[0], punto[1]))
            self.solutions[i] = solution[:, 0]

    def encontrar_vecinos_cercanos(self, punto, numero_de_vecinos=1):
        # Buscar los vecinos más cercanos del punto dado
        distancias, indices = self.arbol.query(punto, k=numero_de_vecinos)

        # Devolver las distancias y los índices de los vecinos más cercanos
        return distancias, indices
    
def dinamica(y,t,g,b):
    x, v = y
    dxdt = v
    dvdt = g - b*v
    return [dxdt, dvdt]

def interpolador(punto, puntos_malla ,t, num_vecinos = 5):

    # Crear una instancia de la clase VecinosCercanos
    Vecinos_interpolador = VecinosCercanos(puntos_malla)

    # Encuentra los vecinos más cercanos al punto arbitrario
    distancias, indices = Vecinos_interpolador.encontrar_vecinos_cercanos(punto, numero_de_vecinos=5)

    # Pesos
    epsilon = 10**(-6)
    pesos = np.zeros(num_vecinos)
    for i in range(num_vecinos):
         pesos[i] = 1/(distancias[i] + epsilon) 
    norma = sum(pesos)
    pesos = pesos/norma

    # Combinacion de soluciones cercanas
    n = len(t)
    interpolacion = np.zeros(n)
    for k in range(num_vecinos):

        # solution = odeint(dinamica, y0, t, args=(puntos_malla[indices[k]][0],puntos_malla[indices[k]][1] ))
        # x = solution[:,0]
        # interpolacion = interpolacion + x * pesos[k]

        solucion = buscador_de_vecinos.solutions[indices[k]]
        interpolacion = interpolacion + solucion*pesos[k]


    return interpolacion

def logposterior(g, b, t, x, sigma = 1, alpha = 100, beta = 10, g_0 = 10, b_0 = 1):
    
        if g>0 and b>0:
            # solution = odeint(dinamica, y0, t, args=(g,b))
            # x_theta = solution[:,0]
            punto = np.array([g,b])
            x_theta = interpolador(punto,puntos_malla, t, num_vecinos= num_vecinos)
            Logf_post = -n*np.log(sigma) - np.sum((x-x_theta)**2) /(2*sigma**2) + (alpha -1)*np.log(g) - alpha*g/g_0 + (beta - 1)*np.log(b) - beta*b/b_0

            return Logf_post
        else:
            Logf_post = -10**100
            return Logf_post 
        
def MetropolisHastingsRW(t_datos,x_datos,inicio,size= 100000,alpha= 100, beta= 10, g_0= 10, b_0= 1):

        # Punto inicial (parametros)
        x = inicio

        sigma1, sigma2 = 0.3, 0.1

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



#######################################
####### Inferencia ####################


# Parametros principales (verdaderos)
g = 5.34
b = 1.15

# Simular las observaciones
n = 31      # Tamaño de muestra (n-1)
cota = 1.5
t = np.linspace(0,cota,num = n)

# ECUACIÓN DIFERENCIAL:
# Condiciones iniciales (posición, velocidad)
y0 = [0.0, 0.0]  

# Soluciones de la ecuación dínamica
solutions = odeint(dinamica, y0 ,t, args=(g,b))

# Coordenadas de caída amortiguada
x = solutions[:,0]
v = solutions[:,1]

# Añadir ruido a los datos
error = norm.rvs(0,0.01,n)
error[0] = 0
x = x + error

# Grafica
# plt.title('Datos simulados con k = %2.2f , b = %2.2f ' % (g, b))
# plt.xlabel('Tiempo')
# plt.ylabel('Posición')
# plt.scatter(t,x, color= 'orange')
# plt.show()

################# Preproceso #################
'Buscador de vecinos cercanos y preproceso para calcular la solucion en de la ecuacion diferencial en cada punto'
inicio = time.time()
# Cantidad de vecinos
num_vecinos = 5

# Definir el dominio de los parametros
g_dom = np.linspace(0, 16, num=10)
b_dom = np.linspace(0, 6, num=10)

# Crear la malla utilizando meshgrid
g_mesh, b_mesh = np.meshgrid(g_dom, b_dom)

# Obtener los puntos de la malla y combinarlos en una matriz
puntos_malla = np.column_stack((g_mesh.ravel(), b_mesh.ravel()))

# Crear una instancia de la clase VecinosCercanos
buscador_de_vecinos = VecinosCercanos(puntos_malla)

# Preproceso, calcula solucion en cada punto de la malla
buscador_de_vecinos.compute_solutions(t, puntos_malla)

##### Visualizacion #####
# Punto arbitrario para encontrar vecinos cercanos
punto_arbitrario = np.array([2.05, 2.88])  

# Encuentra los vecinos más cercanos al punto arbitrario
distancias, indices = buscador_de_vecinos.encontrar_vecinos_cercanos(punto_arbitrario, numero_de_vecinos=num_vecinos)
# # Imprime los resultados
# print("Distancias:", distancias)
# print("Índices de vecinos más cercanos:", indices)

# Gráfica puntos en malla
plt.title('Enmallado')
plt.xlabel('g')
plt.ylabel('b')
plt.scatter(g_mesh,b_mesh)
plt.scatter(punto_arbitrario[0],punto_arbitrario[1])
for k in range(num_vecinos):
    print(puntos_malla[indices[k]])
    plt.scatter(puntos_malla[indices[k]][0],puntos_malla[indices[k]][1], color = 'red')
plt.show()

preproceso = time.time()
print(preproceso - inicio)


#### MCMC propio ########

inicio = np.array([8,3])

# Parametros de distribucion a prioi
g_0 = 5
alpha = 1
b_0 = 2
beta = 1.5
size = 60000


sample = MetropolisHastingsRW(t, x, inicio, size = size, g_0 = g_0, b_0 = b_0, beta = beta, alpha= alpha)
# sample = MetropolisHastingsRW(t, x, inicio, size=size, g_0=g_0, b_0=b_0, beta=beta, alpha=alpha, puntos_malla=puntos_malla)

fin = time.time()
print('Tiempo total: ', fin-inicio)



# # %%
# vecinos = VecinosCercanos(puntos_malla)
# vecinos.compute_solutions(t,puntos_malla)
# solucion = vecinos.solutions
# print(solucion[55])


# print(type(solucion))








# %%
#Visualización
g_sample = sample[:,0]
b_sample = sample[:,1]
log_post = sample[:,2]

burn_in = 5000

plt.title('Cadena')
plt.plot(g_sample[:10000],label = 'g')
plt.plot(b_sample[:10000],label = 'b')
plt.legend()
plt.show()

plt.title('Trayectoria de MCMC')
plt.plot(g_sample,b_sample,linewidth = .5, color = 'gray')
plt.xlabel('g')
plt.ylabel('b')
plt.show() 

plt.title('LogPosterior de la cadena')
plt.plot(log_post, color = 'red')
plt.show()

plt.title('Distribuciones a priori y posterior para g')
plt.hist(g_sample[burn_in:], density= True, bins = 40)
dom_g = np.linspace(0,15,500)
plt.plot(dom_g, gamma.pdf(dom_g, a = alpha , scale = g_0/alpha),color = 'green')
plt.ylabel(r'$f(g)$')
plt.xlabel(r'$g$')
linea = np.linspace(0,0.5, 100)
linea_x = np.ones(100)
plt.plot(linea_x*g, linea, color = 'black', linewidth = 0.5)
plt.show()

plt.title('Distribuciones a priori y posterior para b')
plt.hist(b_sample[burn_in:], density= True, bins = 40)
dom_b = np.linspace(0,8,500)
plt.plot(dom_b, gamma.pdf(dom_b, a = beta, scale = b_0/beta),color = 'green')
plt.ylabel(r'$f(b)$')
plt.xlabel(r'$b$')
linea = np.linspace(0,1, 100)
linea_x = np.ones(100)
plt.plot(linea_x*b, linea, color = 'black', linewidth = 0.5)
plt.show()


# Grafica de la distribución de la curva estimada
t_grafica = np.linspace(0,cota, 100)

space = 4000
# curr = 0
for i in range(0,size ,space):

    submuestreo_g =  g_sample[i-1: i]
    media_g = np.mean(submuestreo_g)

    submuestreo_b =  b_sample[i-1: i]
    media_b = np.mean(submuestreo_b)

    solucion_estimada = odeint(dinamica, y0, t_grafica ,args=(media_g ,media_b))

    x_estimado = solucion_estimada[:,0]

    plt.plot(t_grafica, x_estimado, color = 'purple', alpha = 0.2)


    
plt.scatter(t,x, color= 'green', label = 'Datos sim. k = %2.2f, b = %2.2f' %(g,b))

plt.title('Curva estimada (n = %u)'%(n-1))
plt.xlabel('t')
plt.ylabel('Posición')
plt.legend()
plt.show()