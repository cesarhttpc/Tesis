# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm, gamma

from pytwalk import BUQ

### Example using derived class BUQ
### Define the Forward map with signature F( theta, t)

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
sigma = 0.1 #Stardard dev. for data
error = norm.rvs(0,0.01,n)
error[0] = 0
x_data = x_data + error





### logdensity: log of the density of G, with signature logdensity( data, loc, scale)
### see docstring of BUQ
logdensity=norm.logpdf
simdata = lambda n, loc, scale: norm.rvs( size=n, loc=loc, scale=scale)
par_names=[  r"$g$", r"$b$" ]   #, r"$t_0$"]
g_0 = 10
alpha = 10
b_0 = 2
beta = 1.1
par_prior=[ gamma( alpha, scale = g_0/alpha), gamma(beta, scale=b_0/beta)]
par_supp  = [ lambda g: g>0.0, lambda b: b>0.0]   # , lambda t0: True]
#data = array([3.80951951, 3.94018984, 3.98167993, 3.93859411, 4.10960395])
data = x_data
buq = BUQ( q=3, data=x_data, logdensity=logdensity, sigma=sigma, F=F, t=t, par_names=par_names, par_prior=par_prior, par_supp=par_supp)
# buq.SimData(x = np.array([ 4, 2]))#, 0])) #True parameters alpha=3 lambda=0.1
### The two initial values are simulated from the prior
### redifine buq.SimInit eg.
### buq.SimInit = lambda: array([ 0.001, 1000])+norm.rvs(size=3)*0.001

buq.RunMCMC( T=30_000, burn_in=10000)
buq.PlotPost(par=0, burn_in=10000)
buq.PlotPost(par=1, burn_in=10000) #we may acces the parameters by name also
# print("The twalk sample is available in buq.Ouput: ", buq.Output.shape)
