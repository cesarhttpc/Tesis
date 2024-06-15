# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import gamma, norm
import time
import os
from pytwalk import BUQ

#######################################
###### Funciones ######################

def logistico(P,t,theta_1,theta_2):

    # k = theta_1, L_inf = theta_2
    dPdt = theta_1*(theta_2-P)
    return dPdt

def Forward( theta, t):

    theta_1,theta_2 = theta

    solutions = odeint(dinamica, y0 ,t, args=(theta_1,theta_2))
    x = solutions[:,0]
    return x

# def Forward(theta, t):

#     L_inf, k = theta
#     L = L_inf*(1- np.exp(-k*t))
#     return L

def Metropolis(F, t,y_data, theta1_priori , alpha, theta2_priori , beta, size, Estimar_sigma = False, plot = True):

    if Estimar_sigma == True:
        data_MCMC = y_data
    else:
        data_MCMC = None
    logdensity= norm.logpdf

    simdata = lambda n, loc, scale: norm.rvs( size=n, loc=loc, scale=scale)
    ''' Quitar par_names'''
    par_names=[  r"$g$", r"$b$" ] 

    par_prior=[ gamma( alpha, scale = theta1_priori/alpha), gamma(beta, scale=theta2_priori/beta)]
    par_supp  = [ lambda g: g>0.0, lambda b: b>0.0]

    buq = BUQ( q=3, data=data_MCMC, logdensity=logdensity, sigma = sigma, F=F, t=t, par_names=par_names, par_prior=par_prior, par_supp=par_supp)
    # buq = BUQ( q=2, data=y_data, logdensity=logdensity, sigma = sigma, F=F, t=t, par_names=par_names, par_prior=par_prior, par_supp=par_supp)
    # buq.SimData(x = np.array([ g, b])) #True parameters 


    buq.RunMCMC( T=size, burn_in=10000)

    if plot == True:
        buq.PlotPost(par=0, burn_in=10000)
        buq.PlotPost(par=1, burn_in=10000) 
        buq.PlotCorner(burn_in=10000)
        plt.show()

    return buq.Output   

def visualizacion(monte_carlo,t, burn_in):
        
    theta1_sample_plot = monte_carlo[:,0]
    theta2_sample_plot = monte_carlo[:,1]
    
    # Muestra
    plt.scatter(t,y_data, color = 'orange', label = 'Muestra')
    plt.ylabel('L(t)')
    plt.xlabel('t')
    plt.legend()
    if hacer_reporte == True:
        plt.savefig(path + 'Figuras/Generales/Muestra_'+ modelo + path_sigma + '.png')
    plt.show()
    
    # Distribuciones a priori theta_1 y theta_2
    fig, axs = plt.subplots(1,2,figsize = (10,4))
    fig.suptitle(r'Distribuciones a priori para $k$ y $L_{inf}$')
    quantil1_inf = gamma.ppf(0.001, a = alpha, scale = theta1_priori/alpha)
    quantil1_sup = gamma.ppf(0.999, a = alpha, scale = theta1_priori/alpha)
    t1_quantil = np.linspace(quantil1_inf, quantil1_sup,500)
    axs[0].plot(t1_quantil, gamma.pdf(t1_quantil, a = alpha, scale = theta1_priori/alpha), color = 'green')
    axs[0].set_xlabel(r'$k$')
    # axs[0].set_ylabel(r'$f(\theta_1)$')
    quantil2_inf = gamma.ppf(0.001, a = beta, scale = theta2_priori/beta)
    quantil2_sup = gamma.ppf(0.999, a = beta, scale = theta2_priori/beta)
    t2_quantil = np.linspace(quantil2_inf, quantil2_sup,500)
    axs[1].plot(t2_quantil, gamma.pdf(t2_quantil, a = beta, scale = theta2_priori/beta), color = 'green')
    axs[1].set_xlabel(r'$L_{inf}$')
    # axs[1].set_ylabel(r'$f(\theta_2)$')
    # axs[0].label_outer()
    # axs[1].label_outer()
    if hacer_reporte == True:
        plt.savefig(path + 'Figuras/Generales/Apriori_'+ modelo + path_sigma + '.png')
    plt.show()

    # Posterior Conjunta
    plt.scatter(theta1_sample_plot[burn_in:],theta2_sample_plot[burn_in:],alpha = 0.005, cmap = 'viridis')
    # plt.xlabel('%s'%par_names[0])
    # plt.ylabel('%s'%par_names[1])
    plt.xlabel(r'$k$')
    plt.ylabel(r'$L_{inf}$')
    if hacer_reporte == True:
        plt.savefig(path + 'Figuras/Generales/Conjunta_'+ modelo + path_sigma + '.png')
    plt.show()

    # Posterior Parametro 1
    plt.title(r'Posterior y a priori para $k$')
    plt.hist(theta1_sample_plot[burn_in:], alpha = 0.4, density = True ,bins = 40, label = 'Posterior')
    t_1 = np.linspace(min(theta1_sample_plot[burn_in:]),max(theta1_sample_plot[burn_in:]),500)
    plt.plot(t_1, gamma.pdf(t_1,a = alpha,scale = theta1_priori/alpha),color = 'red', label = 'A priori')
    plt.xlabel(r'$k$')
    plt.xticks(rotation=45, ha='right')
    counts, bin_edges = np.histogram(theta1_sample_plot[burn_in:], bins=40, density=True)
    max_density = np.max(counts)
    # linea = np.linspace(0,max_density/4, 10)
    # linea_x = np.ones(10)
    # plt.plot(linea_x*theta_1, linea, color = 'black', linewidth = 0.5)
    if hacer_reporte == True:
        plt.savefig(path + 'Figuras/Generales/Post_theta1_'+ modelo + path_sigma + '.png')
    plt.legend()
    plt.show()
    
    # Posterior Parametro 2
    plt.title(r'Posterior y a priori para $L_{inf}$')
    plt.hist(theta2_sample_plot[burn_in:], alpha = 0.4, density = True, bins = 40, label = 'Posterior')
    t_2 = np.linspace(min(theta2_sample_plot[burn_in:]),max(theta2_sample_plot[burn_in:]),500)
    plt.plot(t_2, gamma.pdf(t_2, a = beta,scale = theta2_priori/beta),color = 'red',label = 'A priori')
    plt.xlabel(r'$L_{inf}$')
    plt.xticks(rotation=45, ha='right')
    counts, bin_edges = np.histogram(theta2_sample_plot[burn_in:], bins=40, density=True)
    max_density = np.max(counts)
    # linea = np.linspace(0,max_density/4, 10)
    # linea_x = np.ones(10)
    # plt.plot(linea_x*theta_2, linea, color = 'black', linewidth = 0.5)
    if hacer_reporte == True:
        plt.savefig(path + 'Figuras/Generales/Post_theta2_'+ modelo + path_sigma + '.png')
    plt.legend()
    plt.show()
    

    # Distribucion Predictiva
    longitud_fija = []
    plt.title('Distribución predictiva')
    space = 1000
    cota_timpo = 100 
    t_plot = np.linspace(0,cota_tiempo,500)
    for i in range(0,size ,space):

        theta_1_burn = theta1_sample_plot[burn_in:]
        theta_2_burn = theta2_sample_plot[burn_in:]
        
        submuestreo_theta1 =  theta_1_burn[i-1: i]
        media_theta1 = np.mean(submuestreo_theta1)
        submuestreo_theta2 =  theta_2_burn[i-1: i]
        media_theta2 = np.mean(submuestreo_theta2)

        solucion_estimada = odeint(dinamica, y0, t_plot ,args=(media_theta1 ,media_theta2))
        x_estimado = solucion_estimada[:,0]
        longitud_fija.append(x_estimado[150])
        
        plt.plot(t_plot, x_estimado, color = 'gray', alpha = 0.5)
    solucion = odeint(dinamica, y0, t_plot ,args=(theta_1,theta_2))
    x_exacto = solucion[:,0]
    # plt.plot(t_plot,x_exacto,color = 'Blue')
    plt.scatter(t,y_data)
    # plt.scatter(t,x_data, color = 'orange')
    plt.xlabel(r't')
    plt.ylabel(r'L(t)')
    if hacer_reporte == True:
        plt.savefig(path + 'Figuras/Generales/Predictiva_'+ modelo + path_sigma + '.png')
    plt.show()

    # Distribución a un dato
    plt.title('Distribución t = 30')
    plt.hist(longitud_fija, alpha = 0.4, density= True)
    plt.show()


################# Datos Dados ###################


# path_1 = "/EdadLong_1.csv"
path_1 = 'C:/Users/ce_ra/Documents/CIMAT/Semestres/Cuarto/Tesis/Codigo/Consultoria/Peces/EdadLong_01.csv'
# path_2 = "/EdadLong_2.csv"
path_2 = 'C:/Users/ce_ra/Documents/CIMAT/Semestres/Cuarto/Tesis/Codigo/Consultoria/Peces/EdadLong_2.csv'

EdadLong1 = pd.read_csv(path_1)
EdadLong2 = pd.read_csv(path_2)

# datos = EdadLong1.to_numpy()
datos = EdadLong1.to_numpy()
longitud = datos[:,2]
edad = datos[:,1]

y_data = longitud
t = edad


#Graficas de datos
plt.scatter(t,y_data,color='orange')
plt.xlabel('Meses')
plt.ylabel('Longitud')


####### MCMC ######
# Modelo
dinamica = logistico
modelo = 'logistico'

y0 = [1.0]
theta_1 = 0.04
theta_2 = 923

# L_inf = 923
# L_inf = 888


par_names = ['k','L_inf']
theta1_priori = 0.04  
alpha = 1.1
theta2_priori = 923
beta = 1.1

sigma = 1
size = 100000
burn_in = 20000

Estimar_sigma = True
exper_aprox = False
hacer_reporte = False

if Estimar_sigma == True:
    path_sigma ='_sigma'
else:
    path_sigma = ''
    
path = 'Exp_Central_'+ modelo + path_sigma +'/'  # Trayectoria relativa para archivar

if hacer_reporte == True:
    directory = path + 'Figuras/Generales'
    os.makedirs(directory, mode=0o777, exist_ok= True)
    directory = path + 'Figuras/Individual'
    os.makedirs(directory, mode=0o777, exist_ok= True)


############# Experimentos #########
num_vecinos_varios = np.array([5, 8, 16])
num_puntos_malla = np.array([10, 15, 30, 50])

Monte_carlo_aprox_compilador_theta1 = np.zeros( (size,len(num_vecinos_varios)*len(num_puntos_malla)) )
Monte_carlo_aprox_compilador_theta2 = np.zeros( (size,len(num_vecinos_varios)*len(num_puntos_malla)) )
# Tiempo_compilador = []


# Forward Ordinario
inicio_tiempo = time.time()

monte_carlo = Metropolis(F= Forward, t=t, y_data=y_data, theta1_priori = theta1_priori, alpha= alpha, theta2_priori = theta2_priori, beta = beta, size = size, Estimar_sigma = Estimar_sigma ,plot = False)

fin = time.time()
plt.show()

theta1_sample = monte_carlo[:,0]
theta2_sample = monte_carlo[:,1]



#######################
cota_tiempo = 100
visualizacion(monte_carlo,t,burn_in = burn_in)

print(np.mean(theta1_sample))
print(np.mean(theta2_sample))







