# %%
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import odeint


def gravedad(y,t,g,b):
    x, v = y
    dxdt = v
    dvdt = g - b*v
    return [dxdt, dvdt]

def Forward( theta, t):

    g,b = theta

    solutions = odeint(dinamica, y0 ,t, args=(g,b))
    x = solutions[:,0]
    return x

def gravedad_analitica(t,g,b):

    m = 1
    v_Terminal = m*g/b

    x = v_Terminal*(t - m*(1-np.exp(-b*t/m))/b)
    return x


# %%

path = 'Gravedad/'  # Trayectoria relativa para archivar

directory = path 
os.makedirs(directory, mode=0o777, exist_ok= True)

dinamica = gravedad
y0 = [0.0, 0.0]

g = 9.8
b = 5.22
theta = [g,b]

t = np.linspace(0,2,100)
# F = Forward(theta, t)
# plt.plot(t,F)

plt.title('Trayectorias modelo gravitatorio')
g = 9.8
b = 5.22
plt.plot(t,gravedad_analitica(t,g,b), label = 'g = %r, b= %r' %(g,b))

g = 7.8
b = 2.0
plt.plot(t,gravedad_analitica(t,g,b), label = 'g = %r, b= %r' %(g,b))

g = 15
b = 9.3
plt.plot(t,gravedad_analitica(t,g,b), label = 'g = %r, b= %r' %(g,b))

g = 2.8
b = 0.4
plt.plot(t,gravedad_analitica(t,g,b), label = 'g = %r, b= %r' %(g,b))
plt.xlabel(r'$t$')
plt.ylabel(r'$x(t)$')
plt.legend()
# plt.savefig(path + 'Trayectorias.png')

# %%

path = 'Logistico/'
directory = path 
os.makedirs(directory, mode=0o777, exist_ok= True)

def logistico(P,t,K,r):

    dPdt = r*P*(1-P/K)
    return dPdt

plt.title('Trayectorias del modelo logístico')
K = 1325
r = 1.2
theta = [K,r]
t = np.linspace(0,10,500)
dinamica = logistico
y0 = [5]
log = Forward(theta, t)

plt.plot(t,log, label = r'$K = %r, r = %r$'%(K,r))

K = 650
r = 4
theta = [K,r]
t = np.linspace(0,10,500)
dinamica = logistico
y0 = [5]
log = Forward(theta, t)

plt.plot(t,log, label = r'$K = %r, r = %r$'%(K,r))

K = 1200
r = 2.5
theta = [K,r]
t = np.linspace(0,10,500)
dinamica = logistico
y0 = [5]
log = Forward(theta, t)

plt.plot(t,log, label = r'$K = %r, r = %r$'%(K,r))


plt.ylabel(r'$P(t)$')
plt.xlabel(r'$t$')
plt.legend()
# plt.savefig(path + 'Trayectorias_log.png')

# %%


path = 'SIR/'
directory = path 
os.makedirs(directory, mode=0o777, exist_ok= True)

def SIR(y,t,beta,gamma):
    I, S, R= y
    dSdt = -beta*S*I
    dIdt = beta*S*I - gamma*I
    dRdt = gamma*I
    return [dIdt, dSdt, dRdt]

plt.title('Trayectoria del modelo SIR')
beta = 0.00009
gamma = 0.5
t = np.linspace(0,10,500)
y0 = [50,50000,5]
soluciones = odeint(SIR,y0,t,args=(beta,gamma))

I = soluciones[:,0]
S = soluciones[:,1]
R = soluciones[:,2]

plt.plot(t,S, label = 'S')
plt.plot(t,I,label = 'I')
plt.plot(t,R, label = 'R')
plt.ylabel(r'Población')
plt.xlabel(r'$t$')
plt.legend()

# plt.savefig(path + 'Trayectorias_SIR.png')

# %%


from scipy.stats import gamma
t = np.linspace(0,10,500)
alpha  = 2
centro = 5
plt.plot(t,gamma.pdf(t, a = alpha, scale = centro/alpha))