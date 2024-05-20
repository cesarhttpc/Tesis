# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.spatial import cKDTree
from scipy.stats import gamma, norm
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

def Forward_aprox(theta, t):

    # Crear una instancia de la clase VecinosCercanos
    # vecinos_interpolador = VecinosCercanos(puntos_malla)
    ''' Instanciar la clase una vez '''    
    # Encuentra los vecinos más cercanos al punto (g,b)
    distancias, indices = buscador_de_vecinos.encontrar_vecinos_cercanos(theta, numero_de_vecinos=num_vecinos)

    # Pesos
    epsilon = 1e-6 #10**(-6)
    pesos = np.zeros(num_vecinos)
    for i in range(num_vecinos):
         pesos[i] = 1/(distancias[i] + epsilon) 
    norma = sum(pesos)
    pesos = pesos/norma
    
    # Combinacion de soluciones cercanas
    n = len(t)
    interpolacion = np.zeros(n)
    for k in range(num_vecinos):

        solucion = buscador_de_vecinos.solutions[indices[k]]
        interpolacion = interpolacion + solucion*pesos[k]

    return interpolacion 

def Metropolis(F, t,x_data, g_0 , alpha, b_0 , beta, size, plot = True):

    sigma = 0.1 #Stardard dev. for data

    logdensity= norm.logpdf

    simdata = lambda n, loc, scale: norm.rvs( size=n, loc=loc, scale=scale)
    par_names=[  r"$g$", r"$b$" ] 

    par_prior=[ gamma( alpha, scale = g_0/alpha), gamma(beta, scale=b_0/beta)]
    par_supp  = [ lambda g: g>0.0, lambda b: b>0.0]

    buq = BUQ( q=2, data=x_data, logdensity=logdensity, sigma=sigma, F=F, t=t, par_names=par_names, par_prior=par_prior, par_supp=par_supp)
    # buq.SimData(x = np.array([ g, b])) #True parameters 


    buq.RunMCMC( T=size, burn_in=10000)

    if plot == True:
        buq.PlotPost(par=0, burn_in=10000)
        buq.PlotPost(par=1, burn_in=10000) 
        buq.PlotCorner(burn_in=10000)
        plt.show()

    return buq.Output   

def preproceso(puntos_malla):
    'Buscador de vecinos cercanos y preproceso para calcular la solucion en de la ecuacion diferencial en cada punto'

    # Crear una instancia de la clase VecinosCercanos
    buscador_de_vecinos = VecinosCercanos(puntos_malla)

    # Preproceso, calcula solucion en cada punto de la malla
    buscador_de_vecinos.compute_solutions(t, puntos_malla)

    return buscador_de_vecinos

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

'''Aqui cambiar la solución usando la función de la dinamica'''
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





# Parametros de distribucion a prioi (Gamma) y MCMC
g_0 = 10
alpha = 10
b_0 = 2
beta = 1.1
size = 60000
burn_in = 10000
hacer_reporte = True

############# Experimentos #########

num_vecinos_varios = np.array([5, 8, 16])
num_puntos_malla = np.array([10, 15, 30, 50])

Monte_carlo_aprox_compilador_g = np.zeros( (size,len(num_vecinos_varios)*len(num_puntos_malla)) )
Monte_carlo_aprox_compilador_b = np.zeros( (size,len(num_vecinos_varios)*len(num_puntos_malla)) )
# Tiempo_compilador = []


if hacer_reporte == True:
    reporte = open('reporte.txt','w')
    reporte.write('REPORTE DE PROCEDIMIENTO \n\n')
    reporte.write('Se hacen experimentos en inferencia bayesiana de problema inverso con forward map aproximado por vecinos cercanos con distintos vecinos y tamano de malla \n\n')
    reporte.write('Para los vecinos %r \n' %num_vecinos_varios)
    reporte.write('Con las mallas cuadradas %r \n\n' %num_puntos_malla)
    reporte.close()

# Forward Ordinario
inicio_tiempo = time.time()

monte_carlo = Metropolis(F= Forward, t=t, x_data=x_data, g_0 = g_0, alpha= alpha, b_0 = b_0, beta = beta, size = size, plot = False)

fin = time.time()
plt.show()

#Reporte
print('Tiempo Forward Ordinario:')
print('Tiempo total: ', fin-inicio_tiempo)
if hacer_reporte == True:

    reporte = open('reporte.txt','a')
    reporte.write('Forward Ordinario \n')
    reporte.write('  Tiempo %.2f \n\n' %(fin-inicio_tiempo))
    reporte.close()

g_sample = monte_carlo[:,0]
b_sample = monte_carlo[:,1]



contador = 1
for k in range(len(num_vecinos_varios)):

    for j in range(len(num_puntos_malla)):
        
       
        # Monte Carlo con Forward map aproximado
        inicio_tiempo = time.time()

        num_vecinos = num_vecinos_varios[k]
        g_dom = np.linspace(0, 13, num= num_puntos_malla[j])
        b_dom = np.linspace(0, 6, num= num_puntos_malla[j])

        # Crear la malla utilizando meshgrid
        g_mesh, b_mesh = np.meshgrid(g_dom, b_dom)
        # Obtener los puntos de la malla y combinarlos en una matriz
        ''' Hacer las mallas fuera del for '''
        puntos_malla = np.column_stack((g_mesh.ravel(), b_mesh.ravel()))

        ''' Hacer el preproceso fuera de for '''
        buscador_de_vecinos = preproceso(puntos_malla) 
        preproceso_tiempo = time.time()
        
        monte_carlo_aprox = Metropolis(F= Forward_aprox, t=t, x_data=x_data, g_0 = g_0, alpha= alpha, b_0 = b_0, beta = beta, size = size, plot = False)
        Monte_carlo_aprox_compilador_g[:,contador-1] = monte_carlo_aprox[:,0]
        Monte_carlo_aprox_compilador_b[:,contador-1] = monte_carlo_aprox[:,1]

        fin = time.time()
        plt.show()

        # Reporte
        # Tiempo_compilador[contador] = fin-inicio_tiempo
        print('Tiempo Aproximado (vecinos %r):' %num_vecinos_varios[k])
        print('Tiempo preproceso: ', preproceso_tiempo - inicio_tiempo)
        print('Tiempo total: ', fin-inicio_tiempo, '\n')

        if hacer_reporte == True:   
            reporte = open('reporte.txt','a')
            reporte.write('Experimento %r \n'%contador)
            reporte.write('  Forward Aproximado (%r vecinos, %r malla) \n'%(num_vecinos_varios[k],num_puntos_malla[j]))
            reporte.write('  Tiempo: %.2f \n' %(fin-inicio_tiempo))
            reporte.close()

        # visualizacion(monte_carlo_aprox)


        # Graficas
        g_sample_aprox = monte_carlo_aprox[:,0]
        b_sample_aprox = monte_carlo_aprox[:,1]


        plt.title('A priori y posterior para g (%r vecinos y %r malla) '%(num_vecinos_varios[k],num_puntos_malla[j]))
        plt.hist(g_sample[burn_in:], density= True, bins = 20,label = 'Ordinario',alpha = 0.9)
        plt.hist(g_sample_aprox[burn_in:], density=True, bins = 20,label='Aproximado',alpha = 0.8,histtype='step')
        dom_g = np.linspace(8,12,500)
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

        plt.title('A priori y posterior para b (%r vecinos y %r malla) '%(num_vecinos_varios[k],num_puntos_malla[j]))
        plt.hist(b_sample[burn_in:], density= True, bins = 40,label = 'Ordinario',alpha = 0.9)
        plt.hist(b_sample_aprox[burn_in:], density=True, bins = 40, label = 'Aproximado',alpha = 0.8,histtype='step')
        dom_b = np.linspace(0.8,1.6,500)
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




        contador += 1

# # %%

# Grafica convergencia por malla

dom_g = np.linspace(8,12,500)
dom_b = np.linspace(0.8,1.6,500)

for k in range(len(num_vecinos_varios)):

    fig, axs = plt.subplots(1,len(num_vecinos_varios) + 1,figsize = (12,4))
    fig.suptitle('Evolución de la posterior de g para %r vecinos ' %num_vecinos_varios[k])
    for l in range(len(num_vecinos_varios)+1):
        axs[l].hist(g_sample[burn_in:], density = True , bins = 40)
        g_sample_aprox = Monte_carlo_aprox_compilador_g[:,3*k + l]
        axs[l].hist(g_sample_aprox[burn_in:], density = True, bins = 40, histtype  = 'step')
        axs[l].plot(dom_g, gamma.pdf(dom_g, a = alpha , scale = g_0/alpha),color = 'green')
        linea = np.linspace(0,0.3, 100)
        linea_x = np.ones(100)
        axs[l].plot(linea_x*g, linea, color = 'black', linewidth = 0.5)
        # if l != 0 :
        #     axs[l].label_outer()
    if hacer_reporte == True:
        plt.savefig('convergencia_g_%r.png'%k)
    plt.show()


    fig2, axs2 = plt.subplots(1,len(num_vecinos_varios) + 1,figsize = (12,4))
    fig2.suptitle('Evolución de la posterior de b para %r vecinos ' %num_vecinos_varios[k])
    for l in range(len(num_vecinos_varios)+1):
        axs2[l].hist(b_sample[burn_in:], density = True , bins = 40)
        b_sample_aprox = Monte_carlo_aprox_compilador_b[:,3*k + l]
        axs2[l].hist(b_sample_aprox[burn_in:], density = True, bins = 40, histtype  = 'step')
        axs2[l].plot(dom_b, gamma.pdf(dom_b, a = beta, scale = b_0/beta),color = 'green')
        linea = np.linspace(0,1, 100)
        linea_x = np.ones(100)
        axs2[l].plot(linea_x*b, linea, color = 'black', linewidth = 0.5)
        # if l != 0 :
        #     axs[l].label_outer()
    if hacer_reporte == True:
        plt.savefig('convergencia_b_%r.png'%k)
    plt.show()



# # %%

# Grafica de la distribución de la curva estimada
t_grafica = np.linspace(0,cota_dominio, 100)


sol_graf = odeint(dinamica, y0 ,t_grafica, args=(g,b))
x_graf = sol_graf[:,0]

space = 4000
num_experimento = 0
for k in range(len(num_vecinos_varios)):
    
    fig3, axs3 = plt.subplots(1,len(num_vecinos_varios) + 1,figsize = (12,4))
    fig3.suptitle('Distribución de la solución (%r vecinos)'%num_vecinos_varios[k])


    for j in range(len(num_puntos_malla)):

        # plt.title('Curva estimada %r puntos'% num_puntos_malla[j]   )
        for i in range(0,size ,space):

            g_sample_grap = Monte_carlo_aprox_compilador_g[:,num_experimento]
            b_sample_grap = Monte_carlo_aprox_compilador_b[:,num_experimento]
            g_sample_grap = g_sample_grap[burn_in:]
            b_sample_grap = b_sample_grap[burn_in:]
            
            submuestreo_g =  g_sample_grap[i-1: i]
            media_g = np.mean(submuestreo_g)
            submuestreo_b =  b_sample[i-1: i]
            media_b = np.mean(submuestreo_b)

            solucion_estimada = odeint(dinamica, y0, t_grafica ,args=(media_g ,media_b))
            x_estimado = solucion_estimada[:,0]
            
            # plt.plot(t_grafica, x_estimado, color = 'purple', alpha = 0.2)
            axs3[j].plot(t_grafica, x_estimado, color = 'purple', alpha = 0.2)

        axs3[j].plot(t_grafica,x_graf, color = 'blue', linewidth = 0.75)
        num_experimento += 1

        if j != 0 :
            axs3[j].label_outer()
    if hacer_reporte == True:
        plt.savefig('trayectoria_dist_%r.png'%k)
    plt.show()


    # plt.scatter(t,x_data, color= 'blue', label = 'Datos sim. g = %2.2f, b = %2.2f' %(g,b))


    # plt.xlabel('Tiempo')
    # plt.ylabel('Posición')
    # plt.legend()
    # plt.show()

