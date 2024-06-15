# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.spatial import cKDTree
from scipy.stats import gamma, norm
import time
import os
from matplotlib.patches import Rectangle
from pytwalk import BUQ
'''Reviar errores t-student'''
from scipy.stats import t as t_dist

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

def gravedad(y,t,g,b):
    x, v = y
    dxdt = v
    dvdt = g - b*v
    return [dxdt, dvdt]

def logistico(P,t,theta_1,theta_2):

    dPdt = theta_1*P*(theta_2-P)
    return dPdt

def SIR(y,t,beta,gamma):
    I, S, R= y
    dSdt = -beta*S*I
    dIdt = beta*S*I - gamma*I
    dRdt = gamma*I
    return [dIdt, dSdt, dRdt]

def Forward( theta, t):

    theta_1,theta_2 = theta

    solutions = odeint(dinamica, y0 ,t, args=(theta_1,theta_2))
    x = solutions[:,0]
    return x

def Forward_aprox(theta, t):

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
'''
Cosas raras en Estimar_sigma
'''
def Metropolis(F, t,y_data, theta1_priori , alpha, theta2_priori , beta, size, Estimar_sigma = False, plot = True):

    # # data_MCMC = y_data
    # if Estimar_sigma == True:
    #     # sigma = None
    #     data_MCMC = y_data
    # else:
    #     data_MCMC = None




    logdensity= norm.logpdf

    simdata = lambda n, loc, scale: norm.rvs( size=n, loc=loc, scale=scale)
    ''' Quitar par_names'''
    par_names=[  r"$g$", r"$b$" ] 

    par_prior=[ gamma( alpha, scale = theta1_priori/alpha), gamma(beta, scale=theta2_priori/beta)]
    par_supp  = [ lambda g: g>0.0, lambda b: b>0.0]

    buq = BUQ( q=3, data=y_data, logdensity=logdensity, sigma = sigma, F=F, t=t, par_names=par_names, par_prior=par_prior, par_supp=par_supp)
    # buq = BUQ( q=2, data=y_data, logdensity=logdensity, sigma = sigma, F=F, t=t, par_names=par_names, par_prior=par_prior, par_supp=par_supp)
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

#########################
#########################
#########################
current_directory = os.getcwd()
print(f"The new working directory is: {current_directory}")
path_directorio = 'C:/Users/ce_ra/Documents/CIMAT/Semestres/Cuarto/Tesis/Version11/'

Estimar_sigma = True
exper_aprox = True
hacer_reporte = True
GuardarCadena = True

#######################################
####### Inferencia ####################

# modelo = ['gravedad', 'logistico', 'SIR']
dinamica = gravedad
modelo = 'gravedad'
# dinamica = logistico
# modelo = 'logistico'
# dinamica = SIR
# modelo = 'SIR'

# Simular las observaciones
n = 10      # Tamaño de muestra (n-1)

if modelo == 'gravedad':
    # Parametros principales (verdaderos)
    par_names = ['g','b']
    theta_1 = 9.81
    theta_2 = 1.15

    # Muestra
    y0 = [0.0, 0.0]
    cota_tiempo = 1.5
    sigma = 0.1

    # Aproximación (Enmallado)
    theta1_cota_min = 0
    theta1_cota_max = 13
    theta2_cota_min = 0
    theta2_cota_max = 6
if modelo == 'logistico':
    # Parametros principales (verdaderos)
    par_names = ['theta_1','theta_2']
    theta_1 = 0.001
    theta_2 = 1000

    # Muestra
    y0 = [100.0]
    cota_tiempo = 10
    sigma = 30

    # Aproximación (Enmallado)
    theta1_cota_min = 0.0005
    theta1_cota_max = 0.0015
    theta2_cota_min = 980
    theta2_cota_max = 1060
if modelo == 'SIR':
    # Parametros principales (verdaderos)
    par_names = ['beta','gamma']
    theta_1 = 0.009 #0.00009
    theta_2 = 0.5

    # Muestra
    y0 = [50, 500, 5]
    # y0 = [50, 50000, 5]
    cota_tiempo = 10
    sigma = 10
    # sigma = 1000

    # Aproximación (Enmallado)
    theta1_cota_min = 0.0
    theta1_cota_max = 0.08
    theta2_cota_min = 0.1
    theta2_cota_max = 1.5

t = np.linspace(0,cota_tiempo,num = n)

# Muestra (simulacion)
y_data = Forward(np.array([theta_1,theta_2]), t)

# Añadir ruido a los datos
error = norm.rvs(0,sigma,n)
error[0] = 0
y_data = y_data + error
''' error = t_dist.rvs(3,size = n)'''


# Parametros de distribucion a prioi (Gamma) y MCMC
if modelo == 'gravedad':
    theta1_priori = 10  
    alpha = 10
    theta2_priori = 2
    beta = 1.1
    size = 600000
    burn_in = 20000
if modelo == 'logistico':
    theta1_priori = 0.001  
    alpha = 1000
    theta2_priori = 1000
    beta = 1000
    size = 500000
    burn_in = 20000
if modelo == 'SIR':
    theta1_priori = 0.01
    alpha = 1
    theta2_priori = 0.5
    beta = 1
    size = 600000
    burn_in = 20000


if Estimar_sigma == True:
    path_sigma ='_sigma'
else:
    path_sigma = ''
    
path = 'Exp_Central_01_'+ modelo + path_sigma +'/'  # Trayectoria relativa para archivar

if hacer_reporte == True:
    directory = path + 'Figuras/Generales'
    os.makedirs(directory, mode=0o777, exist_ok= True)
    directory = path + 'Figuras/Individual'
    os.makedirs(directory, mode=0o777, exist_ok= True)


############# Experimentos #########
num_vecinos_varios = np.array([5, 8, 3])
num_puntos_malla = np.array([10, 15, 30, 50])

## %%
################ Ejecutable ################
############################################
############################################
############################################
Monte_carlo_aprox_compilador_theta1 = np.zeros( (size,len(num_vecinos_varios)*len(num_puntos_malla)) )
Monte_carlo_aprox_compilador_theta2 = np.zeros( (size,len(num_vecinos_varios)*len(num_puntos_malla)) )
Tiempo_compilador = np.array((len(num_vecinos_varios),len(num_puntos_malla)))

if hacer_reporte == True:
    reporte = open(path + 'reporte.txt','w')
    reporte.write('REPORTE DE PROCEDIMIENTO \n\n')
    reporte.write('Enfoque bayesiano al problema inverso \n\n')
    reporte.write('MODELO: %s \n\n' %modelo)
    reporte.write('Estima sigma: %r\n' %Estimar_sigma)
    reporte.write('Observaciones (muestra): \n')
    reporte.write('%s = %r \n' %(par_names[0],theta_1))
    reporte.write('%s = %r \n' %(par_names[1],theta_2))
    reporte.write('n = %r \n' %n)
    reporte.write('tiempo [0,%r] \n\n' %cota_tiempo)
    reporte.write('Parametros de a priori: \n')
    reporte.write('$theta_1 $= %r \n' %theta1_priori)
    reporte.write('alpha = %r \n' %alpha)
    reporte.write('$theta_2 $= %r \n' %theta2_priori)
    reporte.write('beta = %r \n\n' %beta)
    reporte.write('Parametros de MCMC: \n')
    reporte.write('T = %r \n' %size)
    reporte.write('burn in = %r \n\n' %burn_in)
    reporte.write('Para los vecinos %r \n' %num_vecinos_varios)
    reporte.write('Con las mallas cuadradas %r \n\n' %num_puntos_malla)
    reporte.close()

# Forward Ordinario
inicio_tiempo = time.time()

monte_carlo = Metropolis(F= Forward, t=t, y_data=y_data, theta1_priori = theta1_priori, alpha= alpha, theta2_priori = theta2_priori, beta = beta, size = size, Estimar_sigma = Estimar_sigma ,plot = False)

fin = time.time()
plt.show()

#Reporte
print('Tiempo Forward Ordinario:')
print('Tiempo total: ', fin-inicio_tiempo)
if hacer_reporte == True:

    reporte = open(path + 'reporte.txt','a')
    reporte.write('Forward Ordinario \n')
    reporte.write('  Tiempo %.2f \n\n' %(fin-inicio_tiempo))
    reporte.close()

theta1_sample = monte_carlo[:,0]
theta2_sample = monte_carlo[:,1]
log_posterior = monte_carlo[:,2]

if GuardarCadena == True:

    theta1_sample_col = theta1_sample.reshape(-1,1)
    theta2_sample_col = theta2_sample.reshape(-1,1)
    log_posterior_col = log_posterior.reshape(-1,1)

    # Concatenate arrays horizontally
    combined_array = np.hstack((theta1_sample_col, theta2_sample_col, log_posterior_col))
    print(len(combined_array))

    # Save the combined array to a CSV file
    np.savetxt(path + 'Cadena.csv', combined_array, delimiter=",")#, fmt='%d')


# %%
###########################
###########################
###########################
def visualizacion(monte_carlo,t, burn_in, title = True):
    
    theta1_sample_plot = monte_carlo[:,0]
    theta2_sample_plot = monte_carlo[:,1]

    t_plot = np.linspace(0,cota_tiempo,500)

    if modelo == 'gravedad':
        label = r'Distancia $x(t)$'
    elif modelo == 'logistico':
        label = r'Población $X(t)$'
    elif modelo == 'SIR':
        label = r'Infectados $I(t)$'

    # Trayectorias varias
    # beta_1 = 0.01
    # gamma_1 = 0.5
    # path1 = Forward([beta_1,gamma_1],t_plot)
    # plt.plot(t_plot,path1, label=r'$\beta$ = %r, $\gamma$ = %r'%(beta_1,gamma_1))
    # beta_2 = 0.05
    # gamma_2 = 0.4
    # path2 = Forward([beta_2,gamma_2],t_plot)
    # plt.plot(t_plot,path2, label=r'$\beta$ = %r, $\gamma$ = %r'%(beta_2,gamma_2), color = 'midnightblue')
    # beta_3 = 0.007
    # gamma_3 = 0.9
    # path3 = Forward([beta_3,gamma_3],t_plot)
    # plt.plot(t_plot,path3, label=r'$\beta$ = %r, $\gamma$ = %r'%(beta_3,gamma_3), color = 'dodgerblue')
    # plt.xlabel(r'$t$')
    # plt.ylabel('Infectados')
    # plt.legend()
    # plt.show()

    # Muestra
    plt.scatter(t,y_data, color = 'Mediumorchid', label = 'Muestra')
    plt.xlabel(r't')
    plt.ylabel(label)
    plt.legend()
    if hacer_reporte == True:
        plt.savefig(path + 'Figuras/Generales/Muestra_'+ modelo + path_sigma + '.png',dpi=600)
    plt.show()
    
    # Distribuciones a priori theta_1 y theta_2
    fig, axs = plt.subplots(1,2,figsize = (10,4))
    if title == True:
        fig.suptitle(r'Distribuciones a priori para {}  y {}'.format(par_names[0],par_names[1]))
    quantil1_inf = gamma.ppf(0.001, a = alpha, scale = theta1_priori/alpha)
    quantil1_sup = gamma.ppf(0.999, a = alpha, scale = theta1_priori/alpha)
    t1_quantil = np.linspace(quantil1_inf, quantil1_sup,500)
    axs[0].plot(t1_quantil, gamma.pdf(t1_quantil, a = alpha, scale = theta1_priori/alpha), color = 'seagreen')
    axs[0].set_xlabel(r'$\beta$')
    axs[0].set_ylabel('Densidad')
    quantil2_inf = gamma.ppf(0.001, a = beta, scale = theta2_priori/beta)
    quantil2_sup = gamma.ppf(0.999, a = beta, scale = theta2_priori/beta)
    t2_quantil = np.linspace(quantil2_inf, quantil2_sup,500)
    axs[1].plot(t2_quantil, gamma.pdf(t2_quantil, a = beta, scale = theta2_priori/beta), color = 'seagreen')
    axs[1].set_xlabel(r'$\gamma$')
    axs[1].set_ylabel('Densidad')
    # axs[0].label_outer()
    # axs[1].label_outer()
    if hacer_reporte == True:
        plt.savefig(path + 'Figuras/Generales/Apriori_'+ modelo + path_sigma + '.png',dpi=600)
    plt.show()

    # Posterior Conjunta
    if title == True:
        plt.title(r'Posterior conjunta')
    plt.hist2d(theta1_sample[burn_in:], theta2_sample[burn_in:], bins = 40, cmap='Blues')
    plt.xlabel(par_names[0])
    plt.ylabel(par_names[1])
    if hacer_reporte == True:
        plt.savefig(path + 'Figuras/Generales/Conjunta_'+ modelo + path_sigma + '.png',dpi=600)
    plt.show()

    # Posterior Parametro 1
    if title == True:
        plt.title(r'Posterior y a priori para {}'.format(par_names[0]))
    plt.hist(theta1_sample_plot[burn_in:], density = True ,bins = 40, alpha = 0.5)
    t_1 = np.linspace(min(theta1_sample_plot[burn_in:]),max(theta1_sample_plot[burn_in:]),500)
    plt.plot(t_1, gamma.pdf(t_1,a = alpha,scale = theta1_priori/alpha),color = 'seagreen')
    plt.xlabel(par_names[0])
    plt.xticks(rotation=45, ha='right')
    counts, bin_edges = np.histogram(theta1_sample_plot[burn_in:], bins=40, density=True)
    max_density = np.max(counts)
    linea = np.linspace(0,max_density/4, 10)
    linea_x = np.ones(10)
    plt.plot(linea_x*theta_1, linea, color = 'black', linewidth = 1)
    if hacer_reporte == True:
        plt.savefig(path + 'Figuras/Generales/Post_theta1_'+ modelo + path_sigma + '.png', dpi=600, bbox_inches='tight')
    plt.show()
    
    # Posterior Parametro 2
    plt.title(r'Posterior y a priori para {}'.format(par_names[1]))
    plt.hist(theta2_sample_plot[burn_in:], density = True, bins = 40, alpha = 0.5)
    t_2 = np.linspace(min(theta2_sample_plot[burn_in:]),max(theta2_sample_plot[burn_in:]),500)
    plt.plot(t_2, gamma.pdf(t_2, a = beta,scale = theta2_priori/beta),color = 'seagreen')
    plt.xlabel(par_names[1])
    plt.xticks(rotation=45, ha='right')
    counts, bin_edges = np.histogram(theta2_sample_plot[burn_in:], bins=40, density=True)
    max_density = np.max(counts)
    linea = np.linspace(0,max_density/4, 10)
    linea_x = np.ones(10)
    plt.plot(linea_x*theta_2, linea, color = 'black', linewidth = 1)
    if hacer_reporte == True:
        plt.savefig(path + 'Figuras/Generales/Post_theta2_'+ modelo + path_sigma + '.png', dpi=600, bbox_inches='tight')
    plt.show()

    # Posteriores TODAS
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
    ax0.set_title('Posterior conjunta')
    ax0.hist2d(theta1_sample[burn_in:], theta2_sample[burn_in:], bins = 40, cmap='Blues')
    # ax0.legend(prop={'size': 6})
    ax0.set_xlabel(par_names[0])
    ax0.set_ylabel(par_names[1])
    ax2.set_title(r'Posterior {}'.format(par_names[0]))
    ax2.hist(theta1_sample_plot[burn_in:], density = True ,bins = 40, alpha = 0.5)
    t_1 = np.linspace(min(theta1_sample_plot[burn_in:]),max(theta1_sample_plot[burn_in:]),500)
    ax2.plot(t_1, gamma.pdf(t_1,a = alpha,scale = theta1_priori/alpha),color = 'seagreen')
    counts, bin_edges = np.histogram(theta1_sample_plot[burn_in:], bins=40, density=True)
    max_density = np.max(counts)
    linea = np.linspace(0,max_density/4, 10)
    linea_x = np.ones(10)
    ax2.plot(linea_x*theta_1, linea, color = 'black', linewidth = 1)
    ax2.set_xlabel(par_names[0])
    ax2.set_ylabel(r'Freq')
    ax1.set_title(r'Posterior {}'.format(par_names[1]))
    ax1.hist(theta2_sample_plot[burn_in:], density = True, bins = 40, alpha = 0.5,orientation="horizontal")
    t_2 = np.linspace(min(theta2_sample_plot[burn_in:]),max(theta2_sample_plot[burn_in:]),500)
    ax1.plot(gamma.pdf(t_2, a = beta,scale = theta2_priori/beta),t_2, color = 'seagreen')
    counts, bin_edges = np.histogram(theta2_sample_plot[burn_in:], bins=40, density=True)
    max_density = np.max(counts)
    linea = np.linspace(0,max_density/4, 10)
    linea_x = np.ones(10)
    ax1.plot(linea,linea_x*theta_2, color = 'black', linewidth = 1)
    # ax1.set_xlabel(r'$\gamma$')
    # ax1.label_outer()
    # ax2.label_outer()
    ax1.set_ylabel(par_names[1])
    ax1.set_xlabel(r'Freq')
    # ax3.remove()
    # Distribucion Predictiva
    ax3.set_title('Distribución Predictiva')
    space = 1000
    for i in range(0,size ,space):

        theta_1_burn = theta1_sample_plot[burn_in:]
        theta_2_burn = theta2_sample_plot[burn_in:]
        
        submuestreo_theta1 =  theta_1_burn[i-1: i]
        media_theta1 = np.mean(submuestreo_theta1)
        submuestreo_theta2 =  theta_2_burn[i-1: i]
        media_theta2 = np.mean(submuestreo_theta2)

        solucion_estimada = odeint(dinamica, y0, t_plot ,args=(media_theta1 ,media_theta2))
        x_estimado = solucion_estimada[:,0]
        
        ax3.plot(t_plot, x_estimado, color = 'lightsteelblue', alpha = 0.05)
    solucion = odeint(dinamica, y0, t_plot ,args=(theta_1,theta_2))
    x_exacto = solucion[:,0]
    ax3.plot(t_plot,x_exacto,color = 'Blue')
    ax3.plot(t,y_data, 'o', color = 'Darkorchid', markersize = 3)
    ax3.set_xlabel(r'$t$')
    ax3.set_ylabel(label)
    # plt.xlabel(r't')
    # plt.ylabel(r'Infectados')
    plt.tight_layout()
    if hacer_reporte == True:
        plt.savefig(path + 'Figuras/Generales/Resumen_'+ modelo + path_sigma + '.png', dpi=600)
    plt.show()

    # Distribucion Predictiva
    if title == True:
        plt.title('Distribución predictiva')
    longitud_fija_1 = []
    longitud_fija_2 = []
    tiempo_fijo_indices = [50,249]
    space = 1000
    for i in range(0,size ,space):

        theta_1_burn = theta1_sample_plot[burn_in:]
        theta_2_burn = theta2_sample_plot[burn_in:]
        
        submuestreo_theta1 =  theta_1_burn[i-1: i]
        media_theta1 = np.mean(submuestreo_theta1)
        submuestreo_theta2 =  theta_2_burn[i-1: i]
        media_theta2 = np.mean(submuestreo_theta2)

        solucion_estimada = odeint(dinamica, y0, t_plot ,args=(media_theta1 ,media_theta2))
        x_estimado = solucion_estimada[:,0]
        longitud_fija_1.append(x_estimado[tiempo_fijo_indices[0]])
        longitud_fija_2.append(x_estimado[tiempo_fijo_indices[1]])
        
        plt.plot(t_plot, x_estimado, color = 'lightsteelblue', alpha = 0.05)
    solucion = odeint(dinamica, y0, t_plot ,args=(theta_1,theta_2))
    x_exacto = solucion[:,0]
    plt.plot(t_plot,x_exacto,color = 'Blue')
    plt.xlabel(r't')
    plt.ylabel(label)
    plt.plot(t,y_data, 'o', color = 'Darkorchid' )#color = 'Darkorchid')
    if hacer_reporte == True:
        plt.savefig(path + 'Figuras/Generales/Predictiva_'+ modelo + path_sigma + '.png', dpi=600)
    plt.show()


    # Distribución a un dato
    if title == True:
        plt.title('Distribución t = %2.1f'%t_plot[tiempo_fijo_indices[0]])
    plt.hist(longitud_fija_1, alpha = 0.4, density= True)
    plt.xlabel(label)
    if hacer_reporte == True:
        plt.savefig(path + 'Figuras/Individual/TiempoFijo1_'+ modelo + path_sigma + '.png', dpi=600)
    plt.show()

    # Distribución a un dato
    if title == True:
        plt.title('Distribución t = %2.1f'%t_plot[tiempo_fijo_indices[1]])
    plt.hist(longitud_fija_2, alpha = 0.4, density= True)
    plt.xlabel(label)
    if hacer_reporte == True:
        plt.savefig(path + 'Figuras/Individual/TiempoFijo2_'+ modelo + path_sigma + '.png', dpi=600)
    plt.show()


    # Distribuciones a t fijo
    fig, axs = plt.subplots(1,2,figsize = (10,4))
    axs[0].hist(longitud_fija_1,alpha = 0.4, density = True)
    axs[0].set_xlabel(r'$X(t = %2.1f)$'%t_plot[tiempo_fijo_indices[0]])
    axs[0].set_ylabel('Freq')
    axs[1].hist(longitud_fija_2,alpha = 0.4, density = True)
    axs[1].set_xlabel(r'$X(t = %2.1f)$'%t_plot[tiempo_fijo_indices[1]])
    axs[1].set_ylabel('Freq')
    # axs[1].set_ylabel(r'$f(\theta_2)$')
    # axs[0].label_outer()
    # axs[1].label_outer()
    if hacer_reporte == True:
        plt.savefig(path + 'Figuras/Generales/TiempoFijo_'+ modelo + path_sigma + '.png', dpi=600)
    plt.show()

visualizacion(monte_carlo,t,burn_in = burn_in, title = True)

# %%
if exper_aprox == True:
        
    contador = 1
    for k in range(len(num_vecinos_varios)):

        for j in range(len(num_puntos_malla)):
            
        
            # Monte Carlo con Forward map aproximado
            inicio_tiempo = time.time()

            num_vecinos = num_vecinos_varios[k]

            theta1_dom = np.linspace(theta1_cota_min, theta1_cota_max, num= num_puntos_malla[j])
            theta2_dom = np.linspace(theta2_cota_min, theta2_cota_max, num= num_puntos_malla[j])

            # Crear la malla utilizando meshgrid
            theta1_mesh, theta2_mesh = np.meshgrid(theta1_dom, theta2_dom)
            # Obtener los puntos de la malla y combinarlos en una matriz
            ''' Hacer las mallas fuera del for '''
            puntos_malla = np.column_stack((theta1_mesh.ravel(), theta2_mesh.ravel()))

            ''' Hacer el preproceso fuera de for '''
            buscador_de_vecinos = preproceso(puntos_malla) 
            preproceso_tiempo = time.time()
            
            monte_carlo_aprox = Metropolis(F= Forward_aprox, t=t, y_data=y_data, theta1_priori = theta1_priori, alpha= alpha, theta2_priori = theta2_priori, beta = beta, size = size, Estimar_sigma = Estimar_sigma,plot = False)
            Monte_carlo_aprox_compilador_theta1[:,contador-1] = monte_carlo_aprox[:,0]
            Monte_carlo_aprox_compilador_theta2[:,contador-1] = monte_carlo_aprox[:,1]

            fin = time.time()
            plt.show()

            if GuardarCadena == True:

                theta1_sample_col = monte_carlo_aprox[:,0].reshape(-1,1)
                theta2_sample_col = monte_carlo_aprox[:,1].reshape(-1,1)
                log_posterior_col = monte_carlo_aprox[:,2].reshape(-1,1)

                # Concatenate arrays horizontally
                combined_array = np.hstack((theta1_sample_col, theta2_sample_col, log_posterior_col))
                print(len(combined_array))

                # Save the combined array to a CSV file
                np.savetxt(path + 'Cadena_%r.csv'%contador, combined_array, delimiter=",")#, fmt='%d')


            # Reporte
            # Tiempo_compilador[contador] = fin-inicio_tiempo
            print('Tiempo Aproximado (vecinos %r):' %num_vecinos_varios[k])
            print('Tiempo preproceso: ', preproceso_tiempo - inicio_tiempo)
            print('Tiempo total: ', fin-inicio_tiempo, '\n')

            if hacer_reporte == True:   
                reporte = open(path + 'reporte.txt','a')
                reporte.write('Experimento %r \n'%contador)
                reporte.write('  Forward Aproximado (%r vecinos, %r malla) \n'%(num_vecinos_varios[k],num_puntos_malla[j]))
                reporte.write('  Tiempo: %.2f \n' %(fin-inicio_tiempo))
                reporte.close()

            contador += 1

CadenaCSV = pd.read_csv(path_directorio + path + 'Cadena.csv')
monte_carlo = CadenaCSV.to_numpy()

# %%

def visualizacion_aprox(monte_carlo, t, burn_in, title = True):

    theta1_sample = monte_carlo[:,0]
    theta2_sample = monte_carlo[:,1]    


    Monte_carlo_aprox_compilador_theta1 = np.zeros( (size,len(num_vecinos_varios)*len(num_puntos_malla)) )
    Monte_carlo_aprox_compilador_theta2 = np.zeros( (size,len(num_vecinos_varios)*len(num_puntos_malla)) )


    # Limites de graficas
    counts1, bin_edges1 = np.histogram(theta1_sample[burn_in:], bins=40, density=True)
    max1_density = np.max(counts1)
    min1_histograma = np.min(theta1_sample[burn_in:])
    max1_histograma = np.max(theta1_sample[burn_in:])
    counts2, bin_edges2 = np.histogram(theta2_sample[burn_in:], bins=40, density=True)
    max2_density = np.max(counts2)
    min2_histograma = np.min(theta2_sample[burn_in:])
    max2_histograma = np.max(theta2_sample[burn_in:])

    contador = 1
    for u in range(len(num_vecinos_varios)*len(num_puntos_malla)):

        CadenaCSV = pd.read_csv(path_directorio + path + 'Cadena_%r.csv'%contador, header = None)
        monte_carlo_aprox = CadenaCSV.to_numpy()
        theta1_sample_aprox = monte_carlo_aprox[:,0]
        theta2_sample_aprox = monte_carlo_aprox[:,1]
        Monte_carlo_aprox_compilador_theta1[:,u] = theta1_sample_aprox
        Monte_carlo_aprox_compilador_theta2[:,u] = theta2_sample_aprox

        contador += 1


    contador = 1
    for k in range(len(num_vecinos_varios)):
        for j in range(len(num_puntos_malla)):


            # CadenaCSV = pd.read_csv(path_directorio + path + 'Cadena_%r.csv'%contador)
            # monte_carlo_aprox = CadenaCSV.to_numpy()
            theta1_sample_aprox = Monte_carlo_aprox_compilador_theta1[:,contador-1]
            theta2_sample_aprox = Monte_carlo_aprox_compilador_theta2[:,contador-1]

            ## Posterior theta_1 (ord y aprox)
            if title == True:
                plt.title('A priori y posterior para %s (%r vecinos y %r malla) '%(par_names[0],num_vecinos_varios[k],num_puntos_malla[j]))
            plt.hist(theta1_sample[burn_in:], density= True, bins = 40,label = 'Ordinario',alpha = 0.5)
            plt.hist(theta1_sample_aprox[burn_in:], density=True, bins = 40,label='Aproximado',alpha = 0.8,histtype='step', linewidth = 1.2, color = 'maroon')
            t_1 = np.linspace(min(theta1_sample[burn_in:]),max(theta1_sample[burn_in:]),500)
            plt.plot(t_1, gamma.pdf(t_1, a = alpha , scale = theta1_priori/alpha),color = 'seagreen', label = 'A priori')
            plt.ylabel(r'$f(%s)$'%par_names[0])
            plt.xlabel(r'$%s$'%par_names[0])
            # plt.ylabel(r'$f(\theta_1)$')
            # plt.xlabel(r'$\theta_1$')
            plt.xticks(rotation=45, ha='right')
            # Parametro verdadero
            counts, bin_edges = np.histogram(theta1_sample[burn_in:], bins=40, density=True)
            max_density = np.max(counts)
            linea = np.linspace(0,max_density/4, 10)
            linea_x = np.ones(10)
            plt.plot(linea_x*theta_1, linea, color = 'black', linewidth = 0.9)
            plt.ylim((0,1.2*max1_density))
            plt.xlim((0.9*min1_histograma,1.1*max1_histograma))
            plt.legend()
            if hacer_reporte == True:
                plt.savefig(path + 'Figuras/Individual/PostAprox_theta1_%r_'%contador + modelo + path_sigma + '.png', dpi=600, bbox_inches='tight')
                '''
                plt.savefig(f'{path}Figuras/Individual/PostAprox_theta1_{contador}_{modelo}.png')
                '''
            plt.show()


            # Posterior theta_2 (ord y aprox)
            if title == True:
                plt.title('A priori y posterior para %s (%r vecinos y %r malla) '%(par_names[1],num_vecinos_varios[k],num_puntos_malla[j]))
            plt.hist(theta2_sample[burn_in:], density= True, bins = 40,label = 'Ordinario',alpha = 0.5)
            plt.hist(theta2_sample_aprox[burn_in:], density=True, bins = 40, label = 'Aproximado',alpha = 0.8,histtype='step', linewidth = 1.2, color = 'maroon')
            t_2 = np.linspace(min(theta2_sample[burn_in:]),max(theta2_sample[burn_in:]),500)
            plt.plot(t_2, gamma.pdf(t_2, a = beta, scale = theta2_priori/beta),color = 'seagreen',label = 'A priori')
            plt.ylabel(r'$f(%s)$'%par_names[1])
            plt.xlabel(r'$%s$'%par_names[1])
            # plt.ylabel(r'$f(\theta_1)$')
            # plt.xlabel(r'$\theta_1$')
            plt.xticks(rotation=45, ha='right')
            # Parametro verdadero
            counts, bin_edges = np.histogram(theta2_sample[burn_in:], bins=40, density=True)
            max_density = np.max(counts)
            linea = np.linspace(0,max_density/4, 10)
            linea_x = np.ones(10)
            plt.plot(linea_x*theta_2, linea, color = 'black', linewidth = 0.9)
            plt.ylim((0,1.2*max2_density))
            plt.xlim((0.9*min2_histograma,1.1*max2_histograma))
            plt.legend()
            if hacer_reporte == True:
                plt.savefig(path + 'Figuras/Individual/PostAprox_theta2_%r_'%contador + modelo + path_sigma + '.png', dpi=600, bbox_inches='tight')
            plt.show()


            contador += 1

    # Grafica convergencia por malla
    contador_1 = 1
    contador_2 = 1
    for k in range(len(num_vecinos_varios)):

        fig, axs = plt.subplots(1, len(num_puntos_malla), figsize=(14, 4))
        # fig.suptitle('Evolución de la posterior de %s para %r vecinos ' %(par_names[0],num_vecinos_varios[k]))
        for l in range(len(num_puntos_malla)):

            # CadenaCSV = pd.read_csv(path_directorio + path + 'Cadena_%r.csv'%contador)
            # monte_carlo_aprox = CadenaCSV.to_numpy()
            # theta1_sample_aprox = monte_carlo_aprox[:,0]
            # theta2_sample_aprox = monte_carlo_aprox[:,1]
            theta1_sample_aprox = Monte_carlo_aprox_compilador_theta1[:,contador_1-1]
            

            axs[l].hist(theta1_sample[burn_in:], density=True, bins=40, alpha=0.5)
            # g_sample_aprox = Monte_carlo_aprox_compilador_theta1[:, 3 * k + l]
            axs[l].hist(theta1_sample_aprox[burn_in:], density=True, bins=40, histtype='step', linewidth=1.2, color='maroon')
            axs[l].plot(t_1, gamma.pdf(t_1, a=alpha, scale=theta1_priori / alpha), color='seagreen')
            # plt.ylabel(r'$f(%s)$'%par_names[0])
            # plt.xlabel(r'$%s$'%par_names[0])
            # Parametro verdadero
            counts, bin_edges = np.histogram(theta1_sample[burn_in:], bins=40, density=True)
            max_density = np.max(counts)
            linea = np.linspace(0, max_density / 4, 10)
            linea_x = np.ones(10)
            axs[l].plot(linea_x * theta_1, linea, color='black', linewidth=0.5)

            axs[l].set_ylim([0, 1.2*max1_density])
            axs[l].set_xlim([0.9*min1_histograma, 1.1*max1_histograma])
            axs[l].tick_params(axis="x", labelsize=15) 
            axs[l].tick_params(axis="y", labelsize=15) 

            if (l!=0):
                axs[l].set_yticks([])

            contador_1 += 1

        plt.tight_layout()
            

        # Adding a rectangle to the first subplot
        #rect = Rectangle((0, 0), 1, 1, color='red', alpha=0.3)
        rect_width = 0.1
        rect_height = 0.9
        rect_pos = [-0.10, 0.05, rect_width, rect_height]  # [left, bottom, width, height]

        rect_ax = fig.add_axes(rect_pos)
        rect_ax.add_patch(Rectangle((0, 0), 0.8, 0.8, color="white" ,alpha=0.8))
        rect_ax.text(0.5, 0.5, r'{} vecinos para {}'.format(num_vecinos_varios[k],par_names[0]), ha='center', va='center', transform=rect_ax.transAxes, fontsize=18, rotation=90)

        rect_ax.set_xlim(0, 1)
        rect_ax.set_ylim(0, 1)
        rect_ax.axis('off')
        #axs[0].add_patch(rect)

        if hacer_reporte:
            plt.savefig(path + 'Figuras/Generales/Convergencia_theta1_%r_' % (k + 1) + modelo + path_sigma + '.png', dpi=600, bbox_inches='tight')
        plt.show()

        fig2, axs2 = plt.subplots(1, len(num_puntos_malla), figsize=(14, 4))
        # fig2.suptitle('Evolución de la posterior de %s para %r vecinos ' %(par_names[1],num_vecinos_varios[k]))
        for l in range(len(num_puntos_malla)):

            theta2_sample_aprox = Monte_carlo_aprox_compilador_theta2[:,contador_2-1]

            axs2[l].hist(theta2_sample[burn_in:], density=True, bins=40, alpha=0.5)
            # b_sample_aprox = Monte_carlo_aprox_compilador_theta2[:, 3 * k + l]
            axs2[l].hist(theta2_sample_aprox[burn_in:], density=True, bins=40, histtype='step', linewidth=1.2, color='maroon')
            axs2[l].plot(t_2, gamma.pdf(t_2, a=beta, scale=theta2_priori / beta), color='green')
            # Parametro verdadero
            counts, bin_edges = np.histogram(theta2_sample[burn_in:], bins=40, density=True)
            max_density = np.max(counts)
            linea = np.linspace(0, max_density / 4, 10)
            linea_x = np.ones(10)
            axs2[l].plot(linea_x * theta_2, linea, color='black', linewidth=0.5)

            axs2[l].set_ylim([0, 1.2*max2_density])
            axs2[l].set_xlim([0.9*min2_histograma, 1.1*max2_histograma])
            axs2[l].tick_params(axis="x", labelsize=15) 
            axs2[l].tick_params(axis="y", labelsize=15) 
            
            if (l!=0):
                axs2[l].set_yticks([])

            contador_2 += 1
            
        plt.tight_layout()

        # Adding a rectangle to the first subplot
        #rect = Rectangle((0, 0), 1, 1, color='red', alpha=0.3)
        rect_width = 0.1
        rect_height = 0.9
        rect_pos = [-0.10, 0.05, rect_width, rect_height]  # [left, bottom, width, height]

        rect_ax = fig2.add_axes(rect_pos)
        rect_ax.add_patch(Rectangle((0, 0), 0.8, 0.8, color="white" ,alpha=0.8))
        rect_ax.text(0.5, 0.5, r'{} vecinos para {}'.format(num_vecinos_varios[k],par_names[1]), ha='center', va='center', transform=rect_ax.transAxes, fontsize=18, rotation=90)

        rect_ax.set_xlim(0, 1)
        rect_ax.set_ylim(0, 1)
        rect_ax.axis('off')
        #axs[0].add_patch(rect)

        if hacer_reporte:
            plt.savefig(path + 'Figuras/Generales/Convergencia_theta2_%r_' % (k + 1) + modelo + path_sigma + '.png', dpi=600, bbox_inches='tight')
        plt.show()

visualizacion_aprox(monte_carlo, t, burn_in, title = False)





# # %%

# path_directorio = 'Version11/'
# path = 'Exp_01_SIR_sigma/'
# file_name = 'Cadena.csv'

# full_path = os.path.join(path_directorio, path, file_name)

# if os.path.exists(full_path):
#     CadenaCSV = pd.read_csv(full_path)
# else:
#     raise FileNotFoundError(f"The file {full_path} does not exist.")





# ejecucion = 315

# minutos = ejecucion//60
# segundos = ejecucion%60
# print(minutos, segundos)



