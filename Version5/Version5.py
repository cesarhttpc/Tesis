# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.spatial import cKDTree
from scipy.stats import gamma, uniform, norm
import time

from pytwalk import BUQ

#######################################
###### Funciones ######################

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

def F( theta, t):

    g,b = theta

    y0 = [0.0, 0.0]  
    solutions = odeint(dinamica, y0 ,t, args=(g,b))
    x = solutions[:,0]
    return x

def interpolador(punto, puntos_malla ,t, num_vecinos = 5):

    # Crear una instancia de la clase VecinosCercanos
    Vecinos_interpolador = VecinosCercanos(puntos_malla)

    # Encuentra los vecinos más cercanos al punto arbitrario
    distancias, indices = Vecinos_interpolador.encontrar_vecinos_cercanos(punto, numero_de_vecinos=num_vecinos)

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

        solution = odeint(dinamica, y0, t, args=(puntos_malla[indices[k]][0],puntos_malla[indices[k]][1] ))
        x = solution[:,0]
        interpolacion = interpolacion + x * pesos[k]

        # solucion = buscador_de_vecinos.solutions[indices[k]]
        # interpolacion = interpolacion + solucion*pesos[k]

    return interpolacion

def logNormalAprox(g, b, t, x_data, loc = 0, scale = 1, alpha = 100, beta = 10, g_0 = 10, b_0 = 1):
    
        # Forward map aproximado
        # if g>0 and b>0:
        punto = np.array([g,b])
        x_theta = interpolador(punto,puntos_malla, t, num_vecinos= num_vecinos)
        Logf_post = -n*np.log(scale) - np.sum((x_data-x_theta - loc)**2) /(2*scale**2) 

        return Logf_post
        # else:
        #     Logf_post = -10**100
        #     return Logf_post 

def Metropolis(F, t,x_data, g_0 , alpha, b_0 , beta, size, ForwardAprox = False, plot = True):

    sigma = 0.1 #Stardard dev. for data

    logdensity= norm.logpdf

    simdata = lambda n, loc, scale: norm.rvs( size=n, loc=loc, scale=scale)
    par_names=[  r"$g$", r"$b$" ] 

    par_prior=[ gamma( alpha, scale = g_0/alpha), gamma(beta, scale=b_0/beta)]
    par_supp  = [ lambda g: g>0.0, lambda b: b>0.0]

    buq = BUQ( q=3, data=x_data, logdensity=logdensity, simdata=simdata, sigma=sigma, F=F, t=t, par_names=par_names, par_prior=par_prior, par_supp=par_supp)
    buq.SimData(x = np.array([ g, b])) #True parameters 


    buq.RunMCMC( T=size, burn_in=10000, fnam="cadena_posterior.csv")

    if plot == True:
        buq.PlotPost(par=0, burn_in=10000)
        buq.PlotPost(par=1, burn_in=10000) 
        buq.PlotCorner(burn_in=10000)
        plt.show()

    return buq.Output   

def visualizacion(sample, burn_in = 10000):
     
    g_sample = sample[:,0]
    b_sample = sample[:,1]
    log_post = sample[:,2]

    plt.title('Cadena')
    plt.plot(g_sample[:10000],label = 'g')
    plt.plot(b_sample[:10000],label = 'b')
    plt.legend()
    plt.show()

    plt.title('Trayectoria de MCMC')
    plt.plot(g_sample[burn_in:],b_sample[burn_in:],linewidth = .5, color = 'gray')
    plt.xlabel('g')
    plt.ylabel('b')
    plt.show() 

    plt.title('LogPosterior de la cadena')
    plt.plot(log_post[burn_in:], color = 'red')
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
    t_grafica = np.linspace(0,cota_dominio, 100)

    space = 4000
    ##### ARREGLAR BURN_IN ###
    for i in range(0,size ,space):

        submuestreo_g =  g_sample[i-1: i]
        media_g = np.mean(submuestreo_g)

        submuestreo_b =  b_sample[i-1: i]
        media_b = np.mean(submuestreo_b)

        solucion_estimada = odeint(dinamica, y0, t_grafica ,args=(media_g ,media_b))

        x_estimado = solucion_estimada[:,0]

        plt.plot(t_grafica, x_estimado, color = 'purple', alpha = 0.2)
    
    plt.scatter(t,x_data, color= 'blue', label = 'Datos sim. g = %2.2f, b = %2.2f' %(g,b))

    plt.title('Curva estimada (n = %u)'%(n-1))
    plt.xlabel('t')
    plt.ylabel('Posición')
    plt.legend()
    plt.show()


    estimador_g = np.mean(g_sample[burn_in:])    
    estimador_b = np.mean(b_sample[burn_in:])
    print('Estimador de g: ', estimador_g)  
    print('Estimador de b: ', estimador_b)





#######################################
####### Inferencia ####################


# Parametros principales (verdaderos)
g = 9.81
b = 1.15

# Simular las observaciones
n = 31      # Tamaño de muestra (n-1)
cota_dominio = 1.5
t = np.linspace(0,cota_dominio,num = n)

# ECUACIÓN DIFERENCIAL:
# Condiciones iniciales (posición, velocidad)
y0 = [0.0, 0.0]  

# Soluciones de la ecuación dínamica
solutions = odeint(dinamica, y0 ,t, args=(g,b))

# Coordenadas de caída amortiguada
x_data = solutions[:,0]
v_data = solutions[:,1]

# Añadir ruido a los datos
error = norm.rvs(0,0.01,n)
error[0] = 0
x_data = x_data + error


################# Preproceso #################
'Buscador de vecinos cercanos y preproceso para calcular la solucion en de la ecuacion diferencial en cada punto'

inicio_tiempo = time.time()

# Cantidad de vecinos
num_vecinos = 5

# Definir el dominio de los parametros (malla)
g_dom = np.linspace(0, 22, num=30)
b_dom = np.linspace(0, 8, num=30)

# Crear la malla utilizando meshgrid
g_mesh, b_mesh = np.meshgrid(g_dom, b_dom)

# Obtener los puntos de la malla y combinarlos en una matriz
puntos_malla = np.column_stack((g_mesh.ravel(), b_mesh.ravel()))

# Crear una instancia de la clase VecinosCercanos
buscador_de_vecinos = VecinosCercanos(puntos_malla)

# Preproceso, calcula solucion en cada punto de la malla
buscador_de_vecinos.compute_solutions(t, puntos_malla)

preproceso = time.time()

# Parametros de distribucion a prioi
g_0 = 10
alpha = 10
b_0 = 2
beta = 1.1
size = 60000
burn_in = 10000

####### t-walk ########

MonteCarlo = Metropolis(F= F, t=t, x_data=x_data, g_0 = g_0, alpha= alpha, b_0 = b_0, beta = beta, size = size, ForwardAprox= False, plot = True)
fin = time.time()

visualizacion(MonteCarlo, burn_in = burn_in)

print('Tiempo preproceso: ', preproceso - inicio_tiempo)
print('Tiempo total: ', fin-inicio_tiempo)



# %%

type(gamma(3,1))