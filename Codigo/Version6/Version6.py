# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.spatial import cKDTree
from scipy.stats import gamma, norm, uniform
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

def Forward( theta, t):

    g,b = theta

    y0 = [0.0, 0.0]  
    solutions = odeint(dinamica, y0 ,t, args=(g,b))
    x = solutions[:,0]
    return x

'''def Forward_aprox(theta, t):
    pass
    # # Crear una instancia de la clase VecinosCercanos
    # vecinos_interpolador = VecinosCercanos(puntos_malla)
    
    # # Encuentra los vecinos más cercanos al punto (g,b)
    # distancias, indices = vecinos_interpolador.encontrar_vecinos_cercanos(theta, numero_de_vecinos=num_vecinos)

    # # Pesos
    # epsilon = 10**(-6)
    # pesos = np.zeros(num_vecinos)
    # for i in range(num_vecinos):
    #      pesos[i] = 1/(distancias[i] + epsilon) 
    # norma = sum(pesos)
    # pesos = pesos/norma
    
    # # Combinacion de soluciones cercanas
    # n = len(t)
    # interpolacion = np.zeros(n)
    # for k in range(num_vecinos):

    #     solucion = buscador_de_vecinos.solutions[indices[k]]
    #     interpolacion = interpolacion + solucion*pesos[k]

    # return interpolacion '''

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

def Metropolis(F, t,x_data, g_0 , alpha, b_0 , beta, size, plot = True):

    sigma = 0.1 #Stardard dev. for data

    logdensity= norm.logpdf

    simdata = lambda n, loc, scale: norm.rvs( size=n, loc=loc, scale=scale)
    par_names=[  r"$g$", r"$b$" ] 

    par_prior=[ gamma( alpha, scale = g_0/alpha), gamma(beta, scale=b_0/beta)]
    par_supp  = [ lambda g: g>0.0, lambda b: b>0.0]

    buq = BUQ( q=3, data=x_data, logdensity=logdensity, sigma=sigma, F=F, t=t, par_names=par_names, par_prior=par_prior, par_supp=par_supp)
    # buq.SimData(x = np.array([ g, b])) #True parameters 


    buq.RunMCMC( T=size, burn_in=10000, fnam="cadena_posterior.csv")

    if plot == True:
        buq.PlotPost(par=0, burn_in=10000)
        buq.PlotPost(par=1, burn_in=10000) 
        buq.PlotCorner(burn_in=10000)
        plt.show()

    return buq.Output   

def preproceso():
    'Buscador de vecinos cercanos y preproceso para calcular la solucion en de la ecuacion diferencial en cada punto'

    # Crear una instancia de la clase VecinosCercanos
    buscador_de_vecinos = VecinosCercanos(puntos_malla)

    # Preproceso, calcula solucion en cada punto de la malla
    buscador_de_vecinos.compute_solutions(t, puntos_malla)

    ##### Visualizacion #####
    Grafica = False
    if Grafica == True:
        # Punto arbitrario para encontrar vecinos cercanos
        punto_arbitrario = np.array([2.05, 2.88])  

        # Encuentra los vecinos más cercanos al punto arbitrario
        distancias, indices = buscador_de_vecinos.encontrar_vecinos_cercanos(punto_arbitrario, numero_de_vecinos=num_vecinos)

        # Gráfica puntos en malla
        plt.title('Enmallado')
        plt.xlabel('g')
        plt.ylabel('b')
        plt.scatter(g_mesh,b_mesh)
        # plt.scatter(punto_arbitrario[0],punto_arbitrario[1])
        # for k in range(num_vecinos):
        #     print(puntos_malla[indices[k]])
        #     plt.scatter(puntos_malla[indices[k]][0],puntos_malla[indices[k]][1], color = 'red')
        plt.show()

    return puntos_malla, buscador_de_vecinos

def logposterior(g, b, t, x,Forward_aprox_bool = True, sigma = 1, alpha = 100, beta = 10, g_0 = 10, b_0 = 1):
    
        if g>0 and b>0:
            if Forward_aprox_bool == True:

                # Forward map teorico:
                solution = odeint(dinamica, y0, t, args=(g,b))
                x_theta = solution[:,0]

            else:
                # Forward map aproximado:
                punto = np.array([g,b])
                x_theta = interpolador(punto,puntos_malla, t, num_vecinos= num_vecinos)

            Logf_post = -n*np.log(sigma) - np.sum((x-x_theta)**2) /(2*sigma**2) + (alpha -1)*np.log(g) - alpha*g/g_0 + (beta - 1)*np.log(b) - beta*b/b_0

            return Logf_post
        else:
            Logf_post = -10**100
            return Logf_post 
        
def MetropolisHastingsRW(t_datos,x_datos,inicio,Forward_aprox_bool =True,size= 100000,alpha= 100, beta= 10, g_0= 10, b_0= 1):

        # Punto inicial (parametros)
        x = inicio

        sigma1, sigma2 = 0.3, 0.1

        # 
        sample = np.zeros([size,3])
        sample[0,0] = x[0]  
        sample[0,1] = x[1]
        sample[0,2] = logposterior(x[0], x[1], t_datos, x_datos, alpha=alpha, beta = beta, g_0 = g_0, b_0 = b_0, Forward_aprox_bool=Forward_aprox_bool)

        for k in range(size-1):

            # Simulacion de propuesta
            e1 = norm.rvs(0,sigma1)
            e2 = norm.rvs(0,sigma2)
            e = np.array([e1,e2])
            y = x + e   

            # Cadena de Markov
            log_y = logposterior(y[0], y[1], t_datos, x_datos, alpha = alpha, beta = beta, g_0 = g_0, b_0 = b_0, Forward_aprox_bool= Forward_aprox_bool)
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





# Parametros de distribucion a prioi
g_0 = 10
alpha = 10
b_0 = 2
beta = 1.1
size = 60000
burn_in = 10000
hacer_reporte = True

############# Experimentos #########

num_vecinos_varios = np.array([5, 8, 16])
num_puntos_malla = np.array([10, 40, 100])

if hacer_reporte == True:
    reporte = open('reporte.txt','w')
    reporte.write('REPORTE DE PROCEDIMIENTO \n\n')
    reporte.write('Se hacen experimentos en inferencia bayesiana de problema inverso con forward map aproximado por vecinos cercanos con distintos vecinos y tamano de malla \n\n')
    reporte.write('Para los vecinos %r \n' %num_vecinos_varios)
    reporte.write('Con las mallas cuadradas %r \n\n' %num_puntos_malla)
    reporte.close()

contador = 0
Condicion_grafica_malla = 0
for j in range(len(num_puntos_malla)):

    for k in range(len(num_vecinos_varios)):
        
        # Monte Carlo con Forward map aproximado
        inicio_tiempo = time.time()

        num_vecinos = num_vecinos_varios[k]
        g_dom = np.linspace(0, 13, num= num_puntos_malla[j])
        b_dom = np.linspace(0, 6, num= num_puntos_malla[j])

        # Crear la malla utilizando meshgrid
        g_mesh, b_mesh = np.meshgrid(g_dom, b_dom)
        # Obtener los puntos de la malla y combinarlos en una matriz
        puntos_malla = np.column_stack((g_mesh.ravel(), b_mesh.ravel()))

        # Visualizar la malla
        if Condicion_grafica_malla == 0:
            plt.title('Enmallado (%r,%r)'%(num_puntos_malla[j],num_puntos_malla[j]))
            plt.scatter(g_mesh,b_mesh) 
            plt.xlabel('g')
            plt.ylabel('b')
            plt.show()
            Condicion_grafica_malla = 1

        puntos_malla, buscador_de_vecinos = preproceso()
        preproceso_tiempo = time.time()
        
        inicio = np.array([uniform.rvs(0,10),uniform.rvs(0,7)])
        monte_carlo_aprox = MetropolisHastingsRW(t, x_data, inicio, size = size, g_0 = g_0, b_0 = b_0, beta = beta, alpha= alpha)
        # monte_carlo_aprox = Metropolis(F= Forward_aprox, t=t, x_data=x_data, g_0 = g_0, alpha= alpha, b_0 = b_0, beta = beta, size = size, plot = False)

        #Reporte
        fin = time.time()
        print('Tiempo Aproximado (vecinos %r):' %num_vecinos_varios[k])
        print('Tiempo preproceso: ', preproceso_tiempo - inicio_tiempo)
        print('Tiempo total: ', fin-inicio_tiempo, '\n')

        if hacer_reporte == True:   
            contador += 1
            reporte = open('reporte.txt','a')
            reporte.write('Experimento %r \n'%contador)
            reporte.write('  Forward Aproximado (%r vecinos, %r malla) \n'%(num_vecinos_varios[k],num_puntos_malla[j]))
            reporte.write('  Tiempo: %.2f \n' %(fin-inicio_tiempo))
            reporte.close()

        # visualizacion(monte_carlo_aprox)

        # Forward Exacto
        inicio_tiempo = time.time()

        monte_carlo = Metropolis(F= Forward, t=t, x_data=x_data, g_0 = g_0, alpha= alpha, b_0 = b_0, beta = beta, size = size, plot = False)

        fin = time.time()
        plt.show()


        #Reporte
        print('Tiempo NO Aproximado:')
        print('Tiempo total: ', fin-inicio_tiempo)
        if hacer_reporte == True:

            reporte = open('reporte.txt','a')
            reporte.write('  Forward No Aproximado \n')
            reporte.write('  Tiempo %.2f \n\n' %(fin-inicio_tiempo))
            reporte.close()


        # Graficas
        g_sample_aprox = monte_carlo_aprox[:,0]
        b_sample_aprox = monte_carlo_aprox[:,1]
        g_sample = monte_carlo[:,0]
        b_sample = monte_carlo[:,1]

        plt.title('Priori y posterior para g (%r vecinos y %r malla) '%(num_vecinos_varios[k],num_puntos_malla[j]))
        plt.hist(g_sample_aprox[burn_in:], density=True, bins = 20,label='Aproximado',alpha = 0.8)
        plt.hist(g_sample[burn_in:], density= True, bins = 20,label = 'Exacto',alpha = 0.8)
        dom_g = np.linspace(0,20,500)
        plt.plot(dom_g, gamma.pdf(dom_g, a = alpha , scale = g_0/alpha),color = 'green')
        plt.ylabel(r'$f(g)$')
        plt.xlabel(r'$g$')
        linea = np.linspace(0,0.3, 100)
        linea_x = np.ones(100)
        plt.plot(linea_x*g, linea, color = 'black', linewidth = 0.5)
        plt.legend()
        if hacer_reporte == True:
            plt.savefig('posterior_g_%r.png'%contador)
        plt.show()

        plt.title('Priori y posterior para b (%r vecinos y %r malla) '%(num_vecinos_varios[k],num_puntos_malla[j]))
        plt.hist(b_sample_aprox[burn_in:], density=True, bins = 40, label = 'Aproximado',alpha = 0.8)
        plt.hist(b_sample[burn_in:], density= True, bins = 40,label = 'Exacto',alpha = 0.8)
        dom_b = np.linspace(0,3,500)
        plt.plot(dom_b, gamma.pdf(dom_b, a = beta, scale = b_0/beta),color = 'green')
        plt.ylabel(r'$f(b)$')
        plt.xlabel(r'$b$')
        linea = np.linspace(0,1, 100)
        linea_x = np.ones(100)
        plt.plot(linea_x*b, linea, color = 'black', linewidth = 0.5)
        plt.legend()
        if hacer_reporte == True:
            plt.savefig('posterior_b_%r.png'%contador)
        plt.show()


    Condicion_grafica_malla = 0

